"""Closed-form output-mean recalibration for SALT-Q's non-salient zero-point.

WHY THIS EXISTS. The INT2 2x2 (results_saltq.csv) is strongly super-additive: GPTQ floor 50.35,
z-only 48.54, salient-only (zp_lr=0) 41.55, both 65.77. Each tier ALONE is worse than training
nothing, which means z's role in the joint run is COMPENSATORY: the salient tier's 157.3M weights
move through a 2-bit fakequant every step, the layer's output mean drifts, and z is the only
per-(row, group) freedom that can re-center the other ~98% of the columns. Asking SGD to track
that drift is what put zp_lr on a knife's edge (x5 -> 38.28, /5 -> 63.56, 0 -> 41.55, and the
eps / per-tensor-v experiments both -> ~38): the target moves every step, so any lr either lags
it or amplifies its noise.

This callback removes that job from SGD. The mean drift is COMPUTABLE:

    per row i, drift since capture   c_i = (W_fq,i - W_fq,i^0) . mu_S,      mu_S = E[x_salient]

and z can cancel it exactly in expectation, because the z-term contributes -(z*s) @ pool(x) to the
output, whose mean is -(z*s) @ mu_pool. The minimum-norm coefficient change with exact mean
cancellation is the rank-1 least-squares solution

    d(z*s) = outer(c, mu_pool) / ||mu_pool||^2        ->        recal_dz = d(z*s) / s.

So each recalibration is pure tensor arithmetic (one [out, k] @ [k] matvec + one outer product per
layer); mu_S, mu_pool and the reference row-means are captured ONCE from the first few training
steps via forward pre-hooks. No calibration forward passes during training, no optimizer state
for z (202.4M Adam moments freed), no gradient noise in the compensation.

The result is written into SALTQLinear.recal_dz (LEVEL units), which _z_eff() adds before the
clamp -- so it flows into the forward, the merge-free guarantee, and the deployed export through
the single definition of "the zero-point actually used". In the forward it rides the
separately-cast delta term, so it does NOT suffer the bf16 base+delta ulp swallowing that ate
99.5% of the full-rank z's per-step updates.

WHAT IT DELIBERATELY DOES NOT TOUCH. Only the SALIENT tier's drift is compensated. The zp-LoRA
adapter's own mean shifts are loss-driven task adaptation (job A) and are left alone; cancelling
them would partially neuter the adapter. Task A -> adapter, task B -> this callback, and the two
never fight over the same signal.

DDP. Training runs accelerate DDP with ddp_broadcast_buffers=False, so recal_dz is never synced
by the framework. Replica consistency instead comes from determinism: the captured statistics are
all-reduced ONCE across ranks, after which every recalibration is a pure function of (synced
stats, salient weights) -- and the salient weights are identical on all ranks by DDP's contract.

Gradient checkpointing makes the capture hooks fire again on recompute; sums and counts double
together, so the means are unaffected.
"""

import torch
import torch.distributed as dist
from transformers import TrainerCallback

from .qat_saltq import SALTQLinear


class SALTQMeanRecalibration(TrainerCallback):
    """Capture input means over the first `capture_steps` optimizer steps, then every `interval`
    steps rewrite each layer's recal_dz to cancel the salient tier's output-mean drift."""

    def __init__(self, model, capture_steps: int = 8, interval: int = 50):
        # o_proj (group_k == 0) has no salient tier, hence no drift source: excluded. Symmetric
        # layers have no z to write: excluded (INT2/INT3 runs here are all asymmetric).
        self.layers = [
            m for m in model.modules()
            if isinstance(m, SALTQLinear) and m.group_k > 0 and not m.symmetric
        ]
        self.capture_steps = int(capture_steps)
        self.interval = int(interval)
        self._hooks = []
        self._captured_at = None       # global_step at which stats were finalized
        self._skipped = 0              # layers with degenerate ||mu_pool||

    # ------------------------------------------------------------------ capture

    @staticmethod
    def _pre_hook(mod, inputs):
        x = inputs[0].detach()
        xf = x.reshape(-1, mod.in_features).float()
        if not hasattr(mod, "_recal_n"):
            mod._recal_n = 0
            mod._recal_xsal_sum = torch.zeros(mod.group_k, device=x.device)
            mod._recal_pool_sum = torch.zeros(mod.n_nonsal_g, device=x.device)
        mod._recal_n += xf.shape[0]
        mod._recal_xsal_sum += xf[:, :mod.group_k].sum(0)
        # Same pooling as the forward's correction term: sum within each group, then drop the
        # salient groups (they sit at the front of the permuted layout).
        xp = xf.view(-1, mod.ng, mod.group_size).sum(-1)
        mod._recal_pool_sum += xp[:, mod.n_sal_g:].sum(0)

    def on_train_begin(self, args, state, control, **kwargs):
        for m in self.layers:
            self._hooks.append(m.register_forward_pre_hook(self._pre_hook))
        if state.is_world_process_zero:
            print(f"[SALT-Q][recal] capturing input means over the first {self.capture_steps} "
                  f"steps on {len(self.layers)} layers; recalibrating every {self.interval} steps")

    @torch.no_grad()
    def _finalize_capture(self, state):
        for h in self._hooks:
            h.remove()
        self._hooks = []
        use_dist = dist.is_available() and dist.is_initialized()
        for m in self.layers:
            n = torch.tensor([float(m._recal_n)], device=m._recal_xsal_sum.device)
            if use_dist:
                # Sums and counts reduce together, so unequal per-rank token counts are handled.
                dist.all_reduce(m._recal_xsal_sum)
                dist.all_reduce(m._recal_pool_sum)
                dist.all_reduce(n)
            m._recal_mu_sal = (m._recal_xsal_sum / n).contiguous()
            m._recal_mu_pool = (m._recal_pool_sum / n).contiguous()
            m._recal_pool_norm2 = float(m._recal_mu_pool.dot(m._recal_mu_pool))
            # Reference row-means AT CAPTURE TIME: drift is measured from here, so the few
            # warmup steps before capture ends are treated as part of the reference state.
            m._recal_ref = (m._salient_effective().float() @ m._recal_mu_sal).contiguous()
            m.recal_dz = torch.zeros(
                m.out_features, m.n_nonsal_g, device=m._recal_ref.device, dtype=torch.float32)
            del m._recal_xsal_sum, m._recal_pool_sum, m._recal_n
        self._captured_at = state.global_step
        if state.is_world_process_zero:
            print(f"[SALT-Q][recal] capture finalized at step {state.global_step} "
                  f"({int(n.item())} tokens/layer pooled across ranks)")

    # -------------------------------------------------------------- recalibrate

    @torch.no_grad()
    def _recalibrate(self, state):
        c_abs, clamped, total = [], 0, 0
        for m in self.layers:
            if m._recal_pool_norm2 < 1e-20:
                # A degenerate pooled mean would make the least-squares direction explode.
                # Never observed (RMSNorm outputs are not zero-mean over tokens); skip and count.
                self._skipped += 1
                continue
            c = m._salient_effective().float() @ m._recal_mu_sal - m._recal_ref     # [out]
            dzs = torch.outer(c, m._recal_mu_pool) / m._recal_pool_norm2
            m.recal_dz.copy_(dzs / m._s_eff())
            c_abs.append(c.abs().median())
            z_eff = m.saltq_z.float() + m.recal_dz
            if m.zplora_rank > 0:
                z_eff = z_eff + m._zplora_dz()
            clamped += int(((z_eff < m.Qn) | (z_eff > m.Qp)).sum())
            total += z_eff.numel()
        if state.is_world_process_zero and c_abs:
            med = torch.stack(c_abs).median().item()
            print(f"[SALT-Q][recal] step={state.global_step} median|c|={med:.3e} "
                  f"z-clamp={clamped / max(total, 1):.2%}"
                  + (f" skipped={self._skipped}" if self._skipped else ""))

    # ------------------------------------------------------------------ trigger

    def on_step_end(self, args, state, control, **kwargs):
        if self._captured_at is None:
            if state.global_step >= self.capture_steps:
                self._finalize_capture(state)
        elif (state.global_step - self._captured_at) % self.interval == 0:
            self._recalibrate(state)
