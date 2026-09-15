#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba2
from utils import infer_temporal_branch


class DepthwiseSeparableTemporalConv(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        x = x.transpose(0, 1).unsqueeze(0).contiguous()
        x = self.pointwise(self.depthwise(x))
        return x.squeeze(0).transpose(0, 1).contiguous()


class GatedMotionStem(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size,
        dilation_1,
        dilation_2,
        dropout,
    ):
        super().__init__()
        self.delta_mix = nn.Linear(channels * 2, channels)
        self.motion_1 = DepthwiseSeparableTemporalConv(
            channels, kernel_size, dilation_1
        )
        self.motion_2 = DepthwiseSeparableTemporalConv(
            channels, kernel_size, dilation_2
        )
        self.motion_mix = nn.Linear(channels * 2, channels)
        self.motion_norm = nn.LayerNorm(channels)
        self.motion_gate = nn.Linear(channels * 2, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        sequence = x.transpose(0, 1).unsqueeze(0).contiguous()
        sequence = F.pad(sequence, (1, 1), mode='replicate')
        smooth = F.avg_pool1d(sequence, kernel_size=3, stride=1)
        smooth = smooth.squeeze(0).transpose(0, 1).contiguous()

        if smooth.shape[0] == 1:
            backward_delta = torch.zeros_like(smooth)
            forward_delta = torch.zeros_like(smooth)
        else:
            step = smooth[1:] - smooth[:-1]
            backward_delta = torch.cat(
                [torch.zeros_like(smooth[:1]), step], dim=0
            )
            forward_delta = torch.cat(
                [step, torch.zeros_like(smooth[:1])], dim=0
            )

        delta = self.delta_mix(
            torch.cat([backward_delta, forward_delta], dim=-1)
        )
        motion_1 = F.silu(self.motion_1(delta))
        motion_2 = F.silu(self.motion_2(delta))
        motion = self.motion_mix(
            torch.cat([motion_1, motion_2], dim=-1)
        )
        motion = F.silu(self.motion_norm(motion))
        gate = torch.sigmoid(
            self.motion_gate(torch.cat([x, motion], dim=-1))
        )
        return x + self.dropout(gate * motion)


class BidirectionalMamba2Block(nn.Module):
    def __init__(
        self,
        channels,
        d_state,
        d_conv,
        expand,
        headdim,
        chunk_size,
        ffn_expand,
        dropout,
    ):
        super().__init__()
        self.mamba_norm = nn.LayerNorm(channels)

        forward_mamba = Mamba2(
            d_model=channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            chunk_size=chunk_size,
            use_mem_eff_path=True,
        )
        self.forward_mamba = forward_mamba
        self.backward_mamba = copy.deepcopy(forward_mamba)

        self.direction_gate = nn.Linear(channels * 2, channels)
        self.direction_mix = nn.Linear(channels, channels)
        self.mamba_residual_gate = nn.Parameter(torch.tensor(-1.5))
        nn.init.zeros_(self.direction_gate.weight)
        nn.init.zeros_(self.direction_gate.bias)
        nn.init.eye_(self.direction_mix.weight)
        nn.init.zeros_(self.direction_mix.bias)

        ffn_hidden = channels * ffn_expand
        self.ffn_norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, channels),
        )
        self.ffn_residual_gate = nn.Parameter(torch.tensor(-1.5))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        normalized = self.mamba_norm(x).unsqueeze(0).contiguous()
        forward_state = self.forward_mamba(normalized)

        backward_state = self.backward_mamba(
            normalized.flip(1).contiguous()
        )
        backward_state = backward_state.flip(1).contiguous()

        directions = torch.cat(
            [forward_state, backward_state], dim=-1
        )
        direction_weight = torch.sigmoid(
            self.direction_gate(directions)
        )
        state = (
            direction_weight * forward_state
            + (1.0 - direction_weight) * backward_state
        )
        state = self.direction_mix(state).squeeze(0)

        x = x + torch.sigmoid(
            self.mamba_residual_gate
        ) * self.dropout(state)

        update = self.ffn(self.ffn_norm(x))
        return x + torch.sigmoid(
            self.ffn_residual_gate
        ) * self.dropout(update)


class FeatureAwareTemporalEncoder(nn.Module):
    def __init__(self, input_dim, branch, args):
        super().__init__()
        hidden = args.temporal_hidden

        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden)

        if branch == 'frame':
            self.motion_stem = GatedMotionStem(
                hidden,
                args.motion_kernel,
                args.motion_dilation_1,
                args.motion_dilation_2,
                args.temporal_dropout,
            )
        else:
            self.motion_stem = None

        self.blocks = nn.ModuleList([
            BidirectionalMamba2Block(
                hidden,
                args.mamba_d_state,
                args.mamba_d_conv,
                args.mamba_expand,
                args.mamba_headdim,
                args.mamba_chunk_size,
                args.temporal_ffn_expand,
                args.temporal_dropout,
            )
            for _ in range(args.temporal_layers)
        ])

        self.output_norm = nn.LayerNorm(hidden)
        self.output_proj = nn.Linear(hidden, input_dim)
        self.output_dropout = nn.Dropout(args.temporal_dropout)
        self.adapter_gate = nn.Parameter(torch.tensor(-1.5))

    def forward(self, x):
        encoded = F.gelu(self.input_proj(self.input_norm(x)))

        if self.motion_stem is not None:
            encoded = self.motion_stem(encoded)

        for block in self.blocks:
            encoded = block(encoded)

        update = self.output_proj(self.output_norm(encoded))
        return x + torch.sigmoid(
            self.adapter_gate
        ) * self.output_dropout(update)


class PIVOTMIPL(nn.Module):
    _LAMBERT_ITERS = 8

    def __init__(self, nr_fea, nr_class, args):
        super().__init__()
        self.nr_class = nr_class
        self.num_prototypes = args.num_prototypes
        self.ot_eps = args.ot_eps
        self.occupancy_weight = args.occupancy_weight
        self.time_root_iters = args.time_root_iters
        self.teacher_projection_iters = args.teacher_projection_iters
        self.plan_tol = args.plan_tol
        self.geometry_eps = args.geometry_eps
        self.label_momentum = args.label_momentum

        branch, _ = infer_temporal_branch(args.mat_path)
        self.temporal_encoder = FeatureAwareTemporalEncoder(
            nr_fea, branch, args
        )

        prototypes = torch.randn(
            nr_class, self.num_prototypes, nr_fea
        )
        self.prototypes = nn.Parameter(
            F.normalize(prototypes, dim=-1)
        )

    def _cost(self, x):
        instances = F.normalize(x, dim=-1)
        prototypes = F.normalize(self.prototypes, dim=-1)
        cosine = torch.einsum(
            'ckd,td->ckt', prototypes, instances
        )
        eps = self.geometry_eps
        cosine = cosine.clamp(-1.0 + eps, 1.0 - eps)
        sine = (
            1.0 - cosine.square()
        ).clamp_min(eps * eps).sqrt()
        theta = torch.atan2(sine, cosine)
        return 0.5 * theta.square()

    def _lambert_w_from_log(self, log_x):
        tiny = torch.finfo(log_x.dtype).tiny
        small = log_x < -20.0
        moderate = torch.exp(log_x.clamp(max=1.0))
        large = log_x - torch.log(log_x.clamp_min(1.000001))
        w = torch.where(
            log_x > 1.0, large, moderate
        ).clamp_min(tiny)

        for _ in range(self._LAMBERT_ITERS):
            residual = w + w.log() - log_x
            update = residual * w / (1.0 + w)
            candidate = (w - update).clamp_min(tiny)
            w = torch.where(
                small, torch.exp(log_x), candidate
            ).clamp_min(tiny)

        return w

    def _solve_time_mass(self, local_free_energy):
        length = local_free_energy.numel()
        eps = self.ot_eps
        log_nu = -math.log(length)

        if self.occupancy_weight == 0.0:
            return F.softmax(
                -local_free_energy / eps, dim=0
            )

        output_dtype = local_free_energy.dtype
        energy = local_free_energy.to(torch.float64)

        coefficient = self.occupancy_weight * length / eps
        log_base = (
            math.log(coefficient)
            + log_nu
            - energy / eps
        )

        lower = -torch.logsumexp(
            log_nu - energy / eps, dim=0
        )
        upper = lower + coefficient
        search_upper = (
            upper
            + coefficient
            + (energy.max() - energy.min()).abs() / eps
            + 1.0
        )
        log_scale = lower

        for _ in range(self.time_root_iters):
            w = self._lambert_w_from_log(
                log_base + log_scale
            )
            mass = w / coefficient
            error = mass.sum() - 1.0
            derivative = (
                w / (coefficient * (1.0 + w))
            ).sum().clamp_min(torch.finfo(w.dtype).eps)

            lower = torch.where(
                error <= 0.0,
                torch.maximum(lower, log_scale),
                lower,
            )
            upper = torch.where(
                error > 0.0,
                torch.minimum(upper, log_scale),
                upper,
            )

            newton = log_scale - error / derivative
            midpoint = 0.5 * (lower + upper)
            newton_is_safe = (
                torch.isfinite(newton)
                & (newton > lower)
                & (newton < search_upper)
            )
            log_scale = torch.where(
                newton_is_safe, newton, midpoint
            )

        w = self._lambert_w_from_log(log_base + log_scale)
        time_mass = (w / coefficient).clamp_min(
            torch.finfo(w.dtype).tiny
        )
        time_mass = time_mass / time_mass.sum()
        time_mass = time_mass.to(output_dtype).clamp_min(
            torch.finfo(output_dtype).tiny
        )
        return time_mass / time_mass.sum()

    def _free_occupancy_plan(self, active_cost):
        classes, prototypes, length = active_cost.shape

        log_kernel = -active_cost / self.ot_eps
        log_partition = torch.logsumexp(
            log_kernel.reshape(classes * prototypes, length),
            dim=0,
        )
        log_conditional = (
            log_kernel - log_partition.view(1, 1, length)
        )
        local_free_energy = -self.ot_eps * log_partition
        time_mass = self._solve_time_mass(local_free_energy)

        log_plan = (
            log_conditional
            + time_mass.log().view(1, 1, length)
        )
        plan = log_plan.exp()
        class_mass = plan.sum(dim=(1, 2))

        return {
            'plan': plan,
            'log_plan': log_plan,
            'class_mass': class_mass,
            'time_mass': time_mass,
        }

    def _project_class_time(
        self,
        log_reference,
        class_target,
        time_target,
    ):
        work_dtype = torch.float64
        class_target_64 = class_target.to(work_dtype)
        time_target_64 = time_target.to(work_dtype)

        log_class_target = class_target_64.clamp_min(
            1e-300
        ).log()
        log_time_target = time_target_64.clamp_min(
            1e-300
        ).log()
        log_class_time = torch.logsumexp(
            log_reference, dim=1
        ).to(work_dtype)

        log_u = torch.zeros_like(class_target_64)
        log_v = torch.zeros_like(time_target_64)

        for iteration in range(self.teacher_projection_iters):
            log_u = (
                log_class_target
                - torch.logsumexp(
                    log_class_time + log_v.view(1, -1),
                    dim=1,
                )
            )
            log_v = (
                log_time_target
                - torch.logsumexp(
                    log_class_time + log_u.view(-1, 1),
                    dim=0,
                )
            )

            gauge = (class_target_64 * log_u).sum()
            log_u = log_u - gauge
            log_v = log_v + gauge

            if (
                (iteration + 1) % 5 == 0
                or iteration + 1
                == self.teacher_projection_iters
            ):
                log_class_time_plan = (
                    log_class_time
                    + log_u.view(-1, 1)
                    + log_v.view(1, -1)
                )
                log_class_time_plan = (
                    log_class_time_plan
                    - torch.logsumexp(
                        log_class_time_plan.reshape(-1),
                        dim=0,
                    )
                )
                class_time = log_class_time_plan.exp()
                class_error = (
                    class_time.sum(dim=1) - class_target_64
                ).abs().max()
                time_error = (
                    class_time.sum(dim=0) - time_target_64
                ).abs().max()

                if max(
                    class_error.item(), time_error.item()
                ) <= self.plan_tol:
                    break

        log_scale = (
            log_u.view(-1, 1)
            + log_v.view(1, -1)
        ).to(log_reference.dtype)

        log_teacher = (
            log_reference + log_scale.unsqueeze(1)
        )
        log_teacher = (
            log_teacher
            - torch.logsumexp(
                log_teacher.reshape(-1), dim=0
            )
        )
        teacher = log_teacher.exp()
        return teacher, log_teacher

    def _encode_actor(self, x):
        encoded = self.temporal_encoder(x)
        cost = self._cost(encoded)
        actor = self._free_occupancy_plan(cost)
        logits = actor['class_mass'].clamp_min(1e-30).log()
        return logits, actor, cost

    def forward(self, x):
        logits, _, _ = self._encode_actor(x)
        return logits.view(1, self.nr_class)

    def calculate_objective(self, x, partial_label):
        _, actor, cost = self._encode_actor(x)

        old_target = partial_label.reshape(-1).to(
            device=x.device,
            dtype=actor['plan'].dtype,
        )
        candidate_mask = old_target > 0
        old_target = old_target * candidate_mask
        old_target = (
            old_target
            / old_target.sum().clamp_min(1e-30)
        )

        with torch.no_grad():
            proposal = self._free_occupancy_plan(
                cost.detach()[candidate_mask]
            )

            old_active = old_target[candidate_mask]
            target_active = (
                self.label_momentum * old_active
                + (1.0 - self.label_momentum)
                * proposal['class_mass']
            )
            target_active = (
                target_active
                / target_active.sum().clamp_min(1e-30)
            )

            teacher, log_teacher = self._project_class_time(
                proposal['log_plan'],
                target_active,
                proposal['time_mass'],
            )

            new_target = torch.zeros_like(old_target)
            new_target[candidate_mask] = target_active

        actor_active_log = actor['log_plan'][candidate_mask]
        loss = (
            teacher
            * (log_teacher - actor_active_log)
        ).sum()

        return loss, new_target.view(1, -1).detach(), None

    @torch.no_grad()
    def evaluate_objective(self, x):
        _, actor, _ = self._encode_actor(x)
        return actor['class_mass'].view(1, -1)
