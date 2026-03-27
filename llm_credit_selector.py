from __future__ import print_function

import os
import json
import time
import uuid
import re

try:
    import requests
except ImportError:
    requests = None

# GPTsAPI Chat Completions endpoint (per gateway documentation)
GPTSAPI_CHAT_URL = ""

# ---------- Prompting (JSON-only) ----------

SYSTEM_PROMPT = (
    "You are selecting a credit settlement mode for a multi-agent RL environment.\n"
    "This decision happens ONLY at a TASK COMPLETION event.\n\n"
    "Output requirements:\n"
    "- Output MUST be a SINGLE LINE of valid JSON.\n"
    "- Do NOT wrap in markdown. Do NOT include extra text.\n"
    "- JSON must match the fields and allowed values exactly.\n\n"
    "Decision guidelines:\n"
    "- Choose mode from allowed_modes only.\n"
    "- Prefer stability: avoid redundant or extreme penalties because the env already applies per-step crowd and idle penalties.\n"
    "- If overlap_peak or overlap_avg is high, prefer crowd-focused modes.\n"
    "- If wait_steps is high while overlap is not high, prefer wait-focused modes.\n"
    "- If late stage (remaining_ratio low), prefer modes that help finish tail tasks, but do not invent new modes.\n"
)

REASON_CODES = [
    "HIGH_PEAK_OVERLAP",
    "HIGH_AVG_OVERLAP",
    "HIGH_WAIT",
    "LATE_STAGE",
    "EARLY_STAGE",
    "SCALE_STABILITY",
    "DEFAULT_BASELINE",
]

DEFAULT_MODE_FALLBACK = "crowd_wait_penalty_v1"


def build_choice_schema(allowed_modes):
    """Local schema used for validation (gateway may not support JSON Schema enforcement)."""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "mode": {"type": "string", "enum": list(allowed_modes)},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "reason_code": {"type": "string", "enum": REASON_CODES},
            "note": {"type": "string", "maxLength": 160},
        },
        "required": ["mode", "confidence", "reason_code", "note"],
    }




def _safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default


def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def _extract_chat_content(resp_json):
    try:
        choices = resp_json.get("choices", [])
        if not choices:
            return ""
        msg = choices[0].get("message", {})
        return msg.get("content", "") or ""
    except Exception:
        return ""


def _try_parse_json_object(text):
    if text is None:
        return None

    s = text.strip()
    if not s:
        return None

    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def _validate_choice(choice, schema, allowed_modes):
    if not isinstance(choice, dict):
        return None

    for k in schema.get("required", []):
        if k not in choice:
            return None

    if schema.get("additionalProperties") is False:
        for k in choice.keys():
            if k not in schema["properties"]:
                return None

    mode = str(choice.get("mode", "")).strip()
    if mode not in allowed_modes:
        return None

    try:
        conf = float(choice.get("confidence"))
        if conf < 0.0 or conf > 1.0:
            return None
    except Exception:
        return None

    reason = str(choice.get("reason_code", "")).strip()
    if reason not in REASON_CODES:
        return None

    note = str(choice.get("note", ""))
    maxlen = schema["properties"]["note"].get("maxLength", 160)
    if len(note) > maxlen:
        choice["note"] = note[:maxlen]

    # normalize types
    choice["mode"] = mode
    choice["confidence"] = float(choice["confidence"])
    choice["reason_code"] = reason
    choice["note"] = str(choice["note"])
    return choice


def _discretized_cache_key(payload):
    c = payload.get("ctx", {})
    e = payload.get("env_summary", {})
    key = (
        round(_safe_float(c.get("overlap_avg"), 0.0), 1),
        round(_safe_float(c.get("overlap_peak"), 0.0), 1),
        _safe_int(c.get("wait_steps"), 0) // 10,
        _safe_int(c.get("active_steps"), 0) // 10,
        round(_safe_float(e.get("remaining_ratio"), 1.0), 2),
        (_safe_int(e.get("finisher_idle_streak"), 0) // 5),
    )
    return json.dumps(key, sort_keys=True)


def _compute_remaining_ratio(env):
    num_tasks = _safe_int(getattr(env, "num_tasks", 0), 0)
    if num_tasks <= 0:
        return 1.0

    completed_tasks = getattr(env, "completed_tasks", None)
    if completed_tasks is not None:
        try:
            done = sum(1 for x in completed_tasks if x)
            done = _safe_int(done, 0)
        except Exception:
            done = _safe_int(getattr(env, "ep_tasks_completed", 0), 0)
    else:
        done = _safe_int(getattr(env, "ep_tasks_completed", 0), 0)

    rem = max(num_tasks - done, 0)
    return float(rem) / float(max(num_tasks, 1))


def _get_allowed_modes(env):
    try:
        eng = getattr(env, "credit_engine", None)
        rules = getattr(eng, "_rules", None)
        if isinstance(rules, dict) and len(rules) > 0:
            return list(rules.keys())
    except Exception:
        pass

    return [
        "crowd_wait_penalty_v1",
        "crowd_only_penalty_v1",
        "wait_only_penalty_v1",
        "none",
        "if_tree_demo_v1",
    ]


def _make_payload(ctx, env, allowed_modes):
    finisher_id = _safe_int(getattr(ctx, "finisher_id", 0), 0)

    idle_streak_arr = getattr(env, "idle_streak", None)
    finisher_idle_streak = None
    if idle_streak_arr is not None:
        try:
            finisher_idle_streak = _safe_int(idle_streak_arr[finisher_id], 0)
        except Exception:
            finisher_idle_streak = None

    idle_penalty_coeff = _safe_float(getattr(env, "idle_penalty_coeff", 0.0), 0.0)
    crowd_penalty_coeff = _safe_float(getattr(env, "crowd_penalty_coeff", 0.06), 0.06)

    payload = {
        "allowed_modes": list(allowed_modes),
        "credit_config": dict(getattr(env, "credit_config", {})),
        "ctx": {
            "task_id": _safe_int(getattr(ctx, "task_id", 0), 0),
            "finisher_id": finisher_id,
            "step": _safe_int(getattr(ctx, "step", getattr(env, "ep_step_count", 0)), 0),
            "wait_steps": _safe_int(getattr(ctx, "wait_steps", 0), 0),
            "active_steps": _safe_int(getattr(ctx, "active_steps", 0), 0),
            "overlap_avg": _safe_float(getattr(ctx, "overlap_avg", 0.0), 0.0),
            "overlap_peak": _safe_float(getattr(ctx, "overlap_peak", 0.0), 0.0),
        },
        "env_summary": {
            "num_agents": _safe_int(getattr(env, "num_agents", 0), 0),
            "num_tasks": _safe_int(getattr(env, "num_tasks", 0), 0),
            "remaining_ratio": _compute_remaining_ratio(env),
            "idle_penalty_coeff": idle_penalty_coeff,
            "crowd_penalty_coeff": crowd_penalty_coeff,
            "finisher_idle_streak": finisher_idle_streak,
        },
        "output_format": {
            "mode": "string (one of allowed_modes)",
            "confidence": "float in [0,1]",
            "reason_code": "one of %s" % REASON_CODES,
            "note": "short string <= 160 chars",
        },
        "output_example": {
            "mode": "wait_only_penalty_v1",
            "confidence": 0.74,
            "reason_code": "HIGH_WAIT",
            "note": "Wait dominates while overlap is low; focus on wait signal.",
        },
    }
    return payload


class LLMCreditModeSelector(object):
    def __init__(
        self,
        model="",
        timeout_sec=6.0,
        temperature=0.0,
        max_output_tokens=180,
        min_interval_sec=0.2,
        cache_ttl_sec=2.0,
        debug=False,
        max_retries=0,
        per_episode_budget=40,
    ):
        if requests is None:
            raise RuntimeError("requests is not installed. Run: pip install requests")

        self.model = str(model)
        self.timeout_sec = float(timeout_sec)
        self.temperature = float(temperature)
        self.max_output_tokens = int(max_output_tokens)
        self.min_interval_sec = float(min_interval_sec)
        self.cache_ttl_sec = float(cache_ttl_sec)
        self.debug = bool(debug)
        self.max_retries = int(max_retries)
        self.per_episode_budget = int(per_episode_budget) if per_episode_budget is not None else 0

        self._last_call_t = 0.0
        self._cache = {}  
        self.last_choice = None  
        self._last_mode = DEFAULT_MODE_FALLBACK
        self._ep_seen = None
        self._ep_calls = 0

    def __call__(self, ctx, env):
        allowed_modes = _get_allowed_modes(env)
        payload = _make_payload(ctx, env, allowed_modes)
        key = _discretized_cache_key(payload)
        now = time.time()

        if key in self._cache:
            t0, mode0, meta0 = self._cache[key]
            if now - t0 <= self.cache_ttl_sec:
                self.last_choice = meta0
                return mode0

        if now - self._last_call_t < self.min_interval_sec:
            return str(getattr(env, "credit_mode", self._last_mode))

        if self.per_episode_budget and self.per_episode_budget > 0:
            ep = getattr(env, "episode_idx", None)
            if ep is not None:
                if self._ep_seen != ep:
                    self._ep_seen = ep
                    self._ep_calls = 0
                if self._ep_calls >= self.per_episode_budget:
                    return str(getattr(env, "credit_mode", self._last_mode))


        schema = build_choice_schema(allowed_modes)

        api_key = os.environ.get("API_KEY", "")
        if not api_key:
            if self.debug:
                print("[LLM credit] API_KEY missing -> fallback")
            return str(getattr(env, "credit_mode", DEFAULT_MODE_FALLBACK))

        headers = {
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
            "X-Client-Request-Id": str(uuid.uuid4()),
        }

        # Compose user message: payload + explicit "JSON only"
        user_msg = json.dumps(payload, ensure_ascii=False)
        user_msg += "\n\nReturn ONLY a single-line JSON object with keys: mode, confidence, reason_code, note."

        req_body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
        }
        req_body["extra_body"] = {"enable_thinking": False}
        req_body["stream"] = False


        last_err = None
        for attempt in range(1):
            try:
                r = requests.post(
                    GPTSAPI_CHAT_URL,
                    headers=headers,
                    data=json.dumps(req_body),
                    timeout=self.timeout_sec,
                )

                if self.per_episode_budget and self.per_episode_budget > 0:
                    if getattr(env, "episode_idx", None) is not None:
                        self._ep_calls += 1
                self._last_call_t = time.time()

                if self.debug:
                    print("[LLM credit] HTTP", r.status_code, "attempt", attempt)

                r.raise_for_status()
                resp_json = r.json()

                content = _extract_chat_content(resp_json)
                if self.debug:
                    print("[LLM credit] raw content head:", (content or "")[:200])

                parsed = _try_parse_json_object(content)
                choice = _validate_choice(parsed, schema, allowed_modes)
                if choice is None:
                    last_err = "invalid_choice_or_parse_failed"
                    self._last_call_t = time.time()
                    break

                mode = str(choice.get("mode", DEFAULT_MODE_FALLBACK))
                if mode not in allowed_modes:
                    mode = str(getattr(env, "credit_mode", DEFAULT_MODE_FALLBACK))

                self._last_mode = mode
                self.last_choice = choice
                self._cache[key] = (now, mode, choice)

                if self.debug:
                    print("[LLM credit] choice=", choice)

                return mode

            except Exception as e:
                last_err = repr(e)
                if self.debug:
                    print("[LLM credit] exception:", last_err)
                self._last_call_t = time.time()
                break

        if self.debug:
            print("[LLM credit] fallback after failure:", last_err)
        return str(getattr(env, "credit_mode", self._last_mode))


if __name__ == "__main__":
    class DummyCtx(object):
        task_id = 1
        finisher_id = 0
        step = 10
        wait_steps = 500
        active_steps = 20
        overlap_avg = 0.1
        overlap_peak = 2.0

    class DummyEnv(object):
        num_agents = 4
        num_tasks = 10
        ep_step_count = 10
        completed_tasks = [False] * 10
        idle_penalty_coeff = 0.05
        crowd_penalty_coeff = 0.06
        idle_streak = [3, 0, 1, 2]
        credit_mode = "crowd_wait_penalty_v1"
        credit_config = {"overlap_coeff": 0.2, "wait_coeff": 0.001, "overlap_peak_weight": 0.5}

        class DummyCreditEngine(object):
            _rules = {
                "crowd_wait_penalty_v1": None,
                "crowd_only_penalty_v1": None,
                "wait_only_penalty_v1": None,
                "none": None,
                "if_tree_demo_v1": None,
            }

        credit_engine = DummyCreditEngine()

    sel = LLMCreditModeSelector(debug=True)
    mode = sel(DummyCtx(), DummyEnv())
    print("Selected mode:", mode)
