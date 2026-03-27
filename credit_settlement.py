

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

import math


@dataclass(frozen=True)
class TaskCreditContext:
    task_id: int
    finisher_id: int
    step: int
    first_claim_step: int
    active_steps: int
    overlap_sum: float
    overlap_peak: float

    @property
    def overlap_avg(self) -> float:
        if self.active_steps <= 0:
            return 0.0
        return float(self.overlap_sum) / float(self.active_steps)

    @property
    def wait_steps(self) -> int:
        if self.first_claim_step < 0:
            return 0
        return int(self.step) - int(self.first_claim_step)


class CreditSettlementEngine:


    def __init__(
        self,
        mode: str = "crowd_wait_penalty_v1",
        config: Optional[Dict[str, Any]] = None,
        selector: Optional[Callable[[TaskCreditContext, Any], str]] = None,
    ) -> None:
        self.mode = mode
        self.config: Dict[str, Any] = config or {}
        self.selector = selector


        self._rules: Dict[str, Callable[[Any, TaskCreditContext], Tuple[np.ndarray, Dict[str, Any]]]] = {
            "none": self._rule_none,
            "crowd_wait_penalty_v1": self._rule_crowd_wait_penalty_v1, 
            "crowd_only_penalty_v1": self._rule_crowd_only_penalty_v1,
            "wait_only_penalty_v1": self._rule_wait_only_penalty_v1,
            "crowd_wait_quadratic_v1": self._rule_crowd_wait_quadratic_v1,
            "capped_linear_v1": self._rule_capped_linear_v1,
            "abandon_rescue_bonus_v1": self._rule_abandon_rescue_bonus_v1,
            "stage_adaptive_v1": self._rule_stage_adaptive_v1,
            "peak_sensitive_v1": self._rule_peak_sensitive_v1,
            "efficiency_ratio_v1": self._rule_efficiency_ratio_v1,
            "fast_finish_bonus_v1": self._rule_fast_finish_bonus_v1,
            "late_stage_completion_bonus_v1": self._rule_late_stage_completion_bonus_v1,
            # Team reward
            "team_avg_v1": self._rule_team_avg_v1,
        }

    def settle(self, env: Any, task_id: int, finisher_id: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        ctx = self._make_context(env=env, task_id=task_id, finisher_id=finisher_id)

        mode = self.mode
        if self.selector is not None:
            try:
                mode = str(self.selector(ctx, env))
            except Exception:
                mode = self.mode

        rule = self._rules.get(mode)
        if rule is None:
            mode = "crowd_wait_penalty_v1"
            rule = self._rules[mode]

        credit_vec, meta = rule(env, ctx)
        meta = dict(meta) if meta is not None else {}
        meta.setdefault("mode", mode)
        meta.setdefault("task_id", int(task_id))
        meta.setdefault("finisher_id", int(finisher_id))
        meta.setdefault("task_credit_total", float(np.sum(credit_vec)))
        return credit_vec.astype(np.float32, copy=False), meta
    
    def settle_finisher_scalar(self, env, task_id: int, finisher_id: int):
        credit_vec, meta = self.settle(env=env, task_id=task_id, finisher_id=finisher_id)
        return float(credit_vec[finisher_id]), meta


    def _make_context(self, env: Any, task_id: int, finisher_id: int) -> TaskCreditContext:
        active_steps = int(getattr(env, "task_active_steps")[task_id]) if hasattr(env, "task_active_steps") else 0
        overlap_sum = float(getattr(env, "task_overlap_sum")[task_id]) if hasattr(env, "task_overlap_sum") else 0.0
        overlap_peak = float(getattr(env, "task_overlap_peak")[task_id]) if hasattr(env, "task_overlap_peak") else 0.0
        first_claim_step = int(getattr(env, "task_first_claim_step")[task_id]) if hasattr(env, "task_first_claim_step") else -1
        step = int(getattr(env, "ep_step_count", 0))
        return TaskCreditContext(
            task_id=int(task_id),
            finisher_id=int(finisher_id),
            step=step,
            first_claim_step=first_claim_step,
            active_steps=max(active_steps, 1),
            overlap_sum=overlap_sum,
            overlap_peak=overlap_peak,
        )


    def _rule_none(self, env: Any, ctx: TaskCreditContext) -> Tuple[np.ndarray, Dict[str, Any]]:
        credit = np.zeros(int(getattr(env, "num_agents")), dtype=np.float32)
        return credit, {"detail": "no credit settlement"}

    def _rule_crowd_wait_penalty_v1(self, env: Any, ctx: TaskCreditContext) -> Tuple[np.ndarray, Dict[str, Any]]:
        num_agents = int(getattr(env, "num_agents"))
        credit = np.zeros(num_agents, dtype=np.float32)

        overlap_coeff = float(self.config.get("overlap_coeff", 0.2))
        wait_coeff = float(self.config.get("wait_coeff", 0.001))
        peak_w = float(self.config.get("overlap_peak_weight", 0.5))

        crowd_pen = overlap_coeff * (ctx.overlap_avg + peak_w * float(ctx.overlap_peak))
        wait_pen = wait_coeff * float(ctx.wait_steps)
        task_credit = -(crowd_pen + wait_pen)

        credit[int(ctx.finisher_id)] += float(task_credit)
        return credit, {
            "task_credit_finisher": float(task_credit),
            "crowd_pen": float(crowd_pen),
            "wait_pen": float(wait_pen),
            "wait_steps": int(ctx.wait_steps),
            "overlap_avg": float(ctx.overlap_avg),
            "overlap_peak": float(ctx.overlap_peak),
        }

    def _rule_crowd_only_penalty_v1(self, env: Any, ctx: TaskCreditContext) -> Tuple[np.ndarray, Dict[str, Any]]:
        num_agents = int(getattr(env, "num_agents"))
        credit = np.zeros(num_agents, dtype=np.float32)

        overlap_coeff = float(self.config.get("overlap_coeff", 0.2))
        peak_w = float(self.config.get("overlap_peak_weight", 0.5))
        crowd_pen = overlap_coeff * (ctx.overlap_avg + peak_w * float(ctx.overlap_peak))
        task_credit = -crowd_pen

        credit[int(ctx.finisher_id)] += float(task_credit)
        return credit, {"task_credit_finisher": float(task_credit), "crowd_pen": float(crowd_pen)}

    def _rule_wait_only_penalty_v1(self, env: Any, ctx: TaskCreditContext) -> Tuple[np.ndarray, Dict[str, Any]]:
        num_agents = int(getattr(env, "num_agents"))
        credit = np.zeros(num_agents, dtype=np.float32)

        wait_coeff = float(self.config.get("wait_coeff", 0.001))
        wait_pen = wait_coeff * float(ctx.wait_steps)
        task_credit = -wait_pen

        credit[int(ctx.finisher_id)] += float(task_credit)
        return credit, {"task_credit_finisher": float(task_credit), "wait_pen": float(wait_pen), "wait_steps": int(ctx.wait_steps)}


    def _rule_crowd_wait_quadratic_v1(self, env: Any, ctx: TaskCreditContext) -> Tuple[np.ndarray, Dict[str, Any]]:
        num_agents = int(getattr(env, "num_agents"))
        credit = np.zeros(num_agents, dtype=np.float32)

        overlap_coeff = float(self.config.get("overlap_coeff", 0.2))
        wait_coeff = float(self.config.get("wait_coeff", 0.001))
        peak_w = float(self.config.get("overlap_peak_weight", 0.5))
        p = float(self.config.get("quad_p", 2.0))
        p = max(1.0, p)

        ov_avg_p = math.pow(float(ctx.overlap_avg), p)
        ov_peak_p = math.pow(float(ctx.overlap_peak), p)
        crowd_pen = overlap_coeff * (ov_avg_p + peak_w * ov_peak_p)
        wait_pen = wait_coeff * float(ctx.wait_steps)
        task_credit = -(crowd_pen + wait_pen)

        credit[int(ctx.finisher_id)] += float(task_credit)
        return credit, {
            "task_credit_finisher": float(task_credit),
            "crowd_pen": float(crowd_pen),
            "wait_pen": float(wait_pen),
            "quad_p": float(p),
            "overlap_avg": float(ctx.overlap_avg),
            "overlap_peak": float(ctx.overlap_peak),
            "wait_steps": int(ctx.wait_steps),
        }

    def _rule_capped_linear_v1(self, env: Any, ctx: TaskCreditContext) -> Tuple[np.ndarray, Dict[str, Any]]:

        num_agents = int(getattr(env, "num_agents"))
        credit = np.zeros(num_agents, dtype=np.float32)

        overlap_coeff = float(self.config.get("overlap_coeff", 0.2))
        wait_coeff = float(self.config.get("wait_coeff", 0.001))
        peak_w = float(self.config.get("overlap_peak_weight", 0.5))
        cap = float(self.config.get("credit_cap", 2.0))
        cap = max(0.0, cap)

        crowd_pen = overlap_coeff * (ctx.overlap_avg + peak_w * float(ctx.overlap_peak))
        wait_pen = wait_coeff * float(ctx.wait_steps)
        raw = -(crowd_pen + wait_pen)
        task_credit = max(float(raw), -float(cap))

        credit[int(ctx.finisher_id)] += float(task_credit)
        return credit, {
            "task_credit_finisher": float(task_credit),
            "raw": float(raw),
            "credit_cap": float(cap),
            "crowd_pen": float(crowd_pen),
            "wait_pen": float(wait_pen),
            "overlap_avg": float(ctx.overlap_avg),
            "overlap_peak": float(ctx.overlap_peak),
            "wait_steps": int(ctx.wait_steps),
        }

    def _rule_abandon_rescue_bonus_v1(self, env: Any, ctx: TaskCreditContext) -> Tuple[np.ndarray, Dict[str, Any]]:
        num_agents = int(getattr(env, "num_agents"))
        credit = np.zeros(num_agents, dtype=np.float32)

        overlap_coeff = float(self.config.get("overlap_coeff", 0.2))
        wait_coeff = float(self.config.get("wait_coeff", 0.001))
        peak_w = float(self.config.get("overlap_peak_weight", 0.5))

        rescue_coeff = float(self.config.get("rescue_coeff", 0.004))
        rescue_ov_max = float(self.config.get("rescue_ov_max", 0.5))
        rescue_wait_min = int(self.config.get("rescue_wait_min", 0))
        rescue_bonus_cap = float(self.config.get("rescue_bonus_cap", float("inf")))

        crowd_pen = overlap_coeff * (ctx.overlap_avg + peak_w * float(ctx.overlap_peak))
        wait_pen = wait_coeff * float(ctx.wait_steps)
        abandon = max(0.0, float(ctx.wait_steps) - float(ctx.active_steps))

        grant_bonus = (float(ctx.overlap_avg) <= rescue_ov_max) and (int(ctx.wait_steps) >= rescue_wait_min)
        rescue_bonus = rescue_coeff * abandon if grant_bonus else 0.0
        rescue_bonus = min(rescue_bonus, rescue_bonus_cap)

        task_credit = -(crowd_pen + wait_pen) + float(rescue_bonus)

        credit[int(ctx.finisher_id)] += float(task_credit)
        return credit, {
            "task_credit_finisher": float(task_credit),
            "crowd_pen": float(crowd_pen),
            "wait_pen": float(wait_pen),
            "abandon": float(abandon),
            "rescue_bonus": float(rescue_bonus),
            "grant_bonus": bool(grant_bonus),
            "overlap_avg": float(ctx.overlap_avg),
            "overlap_peak": float(ctx.overlap_peak),
            "wait_steps": int(ctx.wait_steps),
            "active_steps": int(ctx.active_steps),
        }

    def _rule_stage_adaptive_v1(self, env: Any, ctx: TaskCreditContext) -> Tuple[np.ndarray, Dict[str, Any]]:
        num_agents = int(getattr(env, "num_agents"))
        credit = np.zeros(num_agents, dtype=np.float32)

        # remaining_ratio = remaining_tasks / num_tasks
        try:
            completed = getattr(env, "completed_tasks")
            num_tasks = int(getattr(env, "num_tasks"))
            done_cnt = int(np.sum(np.array(completed, dtype=np.int32)))
            remaining_ratio = float(max(num_tasks - done_cnt, 0)) / float(max(num_tasks, 1))
        except Exception:
            remaining_ratio = 0.5

        overlap_coeff = float(self.config.get("overlap_coeff", 0.2))
        wait_coeff = float(self.config.get("wait_coeff", 0.001))
        peak_w = float(self.config.get("overlap_peak_weight", 0.5))

        a0 = float(self.config.get("stage_crowd_a0", 1.0))
        a1 = float(self.config.get("stage_crowd_a1", 1.0))
        b0 = float(self.config.get("stage_wait_b0", 1.0))
        b1 = float(self.config.get("stage_wait_b1", 1.0))

        crowd_scale = a0 + a1 * float(remaining_ratio)
        wait_scale = b0 + b1 * float(1.0 - remaining_ratio)

        crowd_pen = overlap_coeff * crowd_scale * (ctx.overlap_avg + peak_w * float(ctx.overlap_peak))
        wait_pen = wait_coeff * wait_scale * float(ctx.wait_steps)
        task_credit = -(crowd_pen + wait_pen)

        credit[int(ctx.finisher_id)] += float(task_credit)
        return credit, {
            "task_credit_finisher": float(task_credit),
            "crowd_pen": float(crowd_pen),
            "wait_pen": float(wait_pen),
            "remaining_ratio": float(remaining_ratio),
            "crowd_scale": float(crowd_scale),
            "wait_scale": float(wait_scale),
            "overlap_avg": float(ctx.overlap_avg),
            "overlap_peak": float(ctx.overlap_peak),
            "wait_steps": int(ctx.wait_steps),
        }

    def _rule_peak_sensitive_v1(self, env: Any, ctx: TaskCreditContext) -> Tuple[np.ndarray, Dict[str, Any]]:
        num_agents = int(getattr(env, "num_agents"))
        credit = np.zeros(num_agents, dtype=np.float32)

        overlap_coeff = float(self.config.get("overlap_coeff", 0.2))
        peak_coeff = float(self.config.get("peak_coeff", overlap_coeff))
        avg_eps = float(self.config.get("peak_avg_eps", 0.0))
        wait_coeff = float(self.config.get("wait_coeff", 0.001))

        crowd_peak_pen = peak_coeff * float(ctx.overlap_peak)
        crowd_avg_pen = overlap_coeff * avg_eps * float(ctx.overlap_avg)
        wait_pen = wait_coeff * float(ctx.wait_steps)
        task_credit = -(crowd_peak_pen + crowd_avg_pen + wait_pen)

        credit[int(ctx.finisher_id)] += float(task_credit)
        return credit, {
            "task_credit_finisher": float(task_credit),
            "crowd_peak_pen": float(crowd_peak_pen),
            "crowd_avg_pen": float(crowd_avg_pen),
            "wait_pen": float(wait_pen),
            "overlap_avg": float(ctx.overlap_avg),
            "overlap_peak": float(ctx.overlap_peak),
            "wait_steps": int(ctx.wait_steps),
        }

    def _rule_efficiency_ratio_v1(self, env: Any, ctx: TaskCreditContext) -> Tuple[np.ndarray, Dict[str, Any]]:
        num_agents = int(getattr(env, "num_agents"))
        credit = np.zeros(num_agents, dtype=np.float32)

        overlap_coeff = float(self.config.get("overlap_coeff", 0.2))
        peak_w = float(self.config.get("overlap_peak_weight", 0.5))
        eff_coeff = float(self.config.get("eff_ratio_coeff", 0.5))

        ratio = float(ctx.wait_steps) / float(max(int(ctx.active_steps), 1))
        crowd_pen = overlap_coeff * (ctx.overlap_avg + peak_w * float(ctx.overlap_peak))
        eff_pen = eff_coeff * float(ratio)
        task_credit = -(crowd_pen + eff_pen)

        credit[int(ctx.finisher_id)] += float(task_credit)
        return credit, {
            "task_credit_finisher": float(task_credit),
            "crowd_pen": float(crowd_pen),
            "eff_pen": float(eff_pen),
            "ratio": float(ratio),
            "active_steps": int(ctx.active_steps),
            "wait_steps": int(ctx.wait_steps),
        }

    def _rule_fast_finish_bonus_v1(self, env: Any, ctx: TaskCreditContext) -> Tuple[np.ndarray, Dict[str, Any]]:
        num_agents = int(getattr(env, "num_agents"))
        credit = np.zeros(num_agents, dtype=np.float32)

        overlap_coeff = float(self.config.get("overlap_coeff", 0.2))
        wait_coeff = float(self.config.get("wait_coeff", 0.001))
        peak_w = float(self.config.get("overlap_peak_weight", 0.5))
        fast_coeff = float(self.config.get("fast_finish_coeff", 1.0))

        crowd_pen = overlap_coeff * (ctx.overlap_avg + peak_w * float(ctx.overlap_peak))
        wait_pen = wait_coeff * float(ctx.wait_steps)
        bonus = float(fast_coeff) / float(1.0 + float(ctx.wait_steps))
        task_credit = -(crowd_pen + wait_pen) + bonus

        credit[int(ctx.finisher_id)] += float(task_credit)
        return credit, {
            "task_credit_finisher": float(task_credit),
            "crowd_pen": float(crowd_pen),
            "wait_pen": float(wait_pen),
            "fast_bonus": float(bonus),
            "wait_steps": int(ctx.wait_steps),
            "overlap_avg": float(ctx.overlap_avg),
            "overlap_peak": float(ctx.overlap_peak),
        }

    def _rule_late_stage_completion_bonus_v1(self, env: Any, ctx: TaskCreditContext) -> Tuple[np.ndarray, Dict[str, Any]]:
        num_agents = int(getattr(env, "num_agents"))
        credit = np.zeros(num_agents, dtype=np.float32)

        try:
            completed = getattr(env, "completed_tasks")
            num_tasks = int(getattr(env, "num_tasks"))
            done_cnt = int(np.sum(np.array(completed, dtype=np.int32)))
            remaining_ratio = float(max(num_tasks - done_cnt, 0)) / float(max(num_tasks, 1))
        except Exception:
            remaining_ratio = 0.5

        overlap_coeff = float(self.config.get("overlap_coeff", 0.2))
        wait_coeff = float(self.config.get("wait_coeff", 0.001))
        peak_w = float(self.config.get("overlap_peak_weight", 0.5))
        late_coeff = float(self.config.get("late_stage_coeff", 0.5))

        crowd_pen = overlap_coeff * (ctx.overlap_avg + peak_w * float(ctx.overlap_peak))
        wait_pen = wait_coeff * float(ctx.wait_steps)
        late_factor = float(1.0 - remaining_ratio)
        bonus = float(late_coeff) * late_factor
        task_credit = -(crowd_pen + wait_pen) + bonus

        credit[int(ctx.finisher_id)] += float(task_credit)
        return credit, {
            "task_credit_finisher": float(task_credit),
            "crowd_pen": float(crowd_pen),
            "wait_pen": float(wait_pen),
            "late_bonus": float(bonus),
            "remaining_ratio": float(remaining_ratio),
            "wait_steps": int(ctx.wait_steps),
            "overlap_avg": float(ctx.overlap_avg),
            "overlap_peak": float(ctx.overlap_peak),
        }
    
    def _rule_team_avg_v1(self, env: Any, ctx: TaskCreditContext) -> Tuple[np.ndarray, Dict[str, Any]]:
        num_agents = int(getattr(env, "num_agents"))
        credit = np.zeros(num_agents, dtype=np.float32)

        overlap_coeff = float(self.config.get("overlap_coeff", 0.2))
        wait_coeff = float(self.config.get("wait_coeff", 0.001))
        peak_w = float(self.config.get("overlap_peak_weight", 0.5))

        crowd_pen = overlap_coeff * (ctx.overlap_avg + peak_w * float(ctx.overlap_peak))
        wait_pen = wait_coeff * float(ctx.wait_steps)
        task_credit = -(crowd_pen + wait_pen)

        per_agent = float(task_credit) / float(max(num_agents, 1))
        credit[:] += per_agent

        return credit, {
            "task_credit_total": float(task_credit),
            "task_credit_per_agent": float(per_agent),
            "crowd_pen": float(crowd_pen),
            "wait_pen": float(wait_pen),
            "wait_steps": int(ctx.wait_steps),
            "overlap_avg": float(ctx.overlap_avg),
            "overlap_peak": float(ctx.overlap_peak),
        }
