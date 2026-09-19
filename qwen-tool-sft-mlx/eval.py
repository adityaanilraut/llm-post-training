"""Strict tool-call benchmark on MLX (no torch).

Compares base model vs --adapter on the SAME official template:
  tokenizer.apply_chat_template(msgs, tools=TOOLS, add_generation_prompt=True)

Scoring (strict, no fallbacks):
  - <tool_call>{...}</tool_call> required; raw JSON without tags = FAIL
  - score 1.0 name+args exact-ish, 0.5 name-only, 0 else
  - no_tool case: 1.0 iff NO <tool_call> block

Usage:
  python eval.py                          # base only
  python eval.py --adapter ./adapters     # base vs LoRA
  python eval.py --limit 2                # smoke test (2 cases)
"""
from __future__ import annotations
import argparse, gc, json, re, time

TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Get current temperature and conditions for a city.",
        "parameters": {"type": "object",
            "properties": {"city": {"type": "string"},
                           "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
            "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "calculator",
        "description": "Evaluate basic arithmetic.",
        "parameters": {"type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"]}}},
]

CASES = [
    {"id": "weather_sf", "user": "What's the weather in San Francisco right now?",
     "tools": [TOOLS[0]], "name": "get_weather", "args": {"city": "San Francisco"}},
    {"id": "weather_tokyo", "user": "Weather in Tokyo?",
     "tools": TOOLS, "name": "get_weather", "args": {"city": "Tokyo"}},
    {"id": "calc", "user": "Compute 128 multiplied by 6",
     "tools": [TOOLS[1]], "name": "calculator", "args": {"expression": "128 * 6"}},
    {"id": "calc_paren", "user": "What is (14 + 6) * 3?",
     "tools": TOOLS, "name": "calculator", "args": {"expression": "(14 + 6) * 3"}},
    {"id": "pick_tool", "user": "What is 81 divided by 9?",
     "tools": TOOLS, "name": "calculator", "args": {"expression": "81 / 9"}},
    {"id": "no_tool", "user": "Who walked on the moon first?",
     "tools": [TOOLS[0]], "name": None, "args": None},
    {"id": "clarify", "user": "Check the weather please.",
     "tools": [TOOLS[0]], "name": None, "args": None, "must_ask": True},
    {"id": "chit_chat", "user": "Hey, how are you?",
     "tools": TOOLS, "name": None, "args": None},
]

RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def parse(text):
    m = RE.search(text)
    if not m:
        return None, False
    try:
        return json.loads(m.group(1).strip()), True
    except Exception:
        return None, False


def norm(v):
    return v.strip().lower() if isinstance(v, str) else v


def score(text, case):
    parsed, valid = parse(text)
    if case["name"] is None:
        # Strict: ANY tool attempt (even malformed/broken tag) is a FAIL.
        # Old bug: broken "<tool_call>...no </tool_call>" parsed as None -> scored 1.0.
        if "<tool_call" in text or parsed is not None:
            return {"valid": False, "name_ok": False, "fuzzy": 0.0, "score": 0.0, "parsed": parsed}
        if case.get("must_ask"):
            t = text.lower()
            ok = any(w in t for w in ("which city", "which", "city", "where"))
            return {"valid": valid, "name_ok": ok, "fuzzy": 1.0 if ok else 0.0,
                    "score": 1.0 if ok else 0.0, "parsed": None}
        # chit-chat / trivia must NOT ask for a city (old LoRA answered
        # "Which city should I look up?" to "Hey, how are you?" and got 1.0)
        t = text.lower()
        if "which city" in t or "which city should" in t:
            return {"valid": valid, "name_ok": False, "fuzzy": 0.0, "score": 0.0, "parsed": None}
        return {"valid": valid, "name_ok": True, "fuzzy": 1.0, "score": 1.0, "parsed": None}
    if parsed is None:
        return {"valid": False, "name_ok": False, "fuzzy": 0.0, "score": 0.0, "parsed": None}
    name_ok = parsed.get("name") == case["name"]
    args = parsed.get("arguments", {})
    if isinstance(args, str):
        try: args = json.loads(args)
        except Exception: args = {}
    exp = case["args"] or {}
    hits = 0
    for k, ev in exp.items():
        if k not in args: continue
        av = args[k]
        if norm(av) == norm(ev): hits += 1
        elif isinstance(ev, str) and isinstance(av, str) and norm(ev) in norm(av): hits += 0.5
    fuzzy = hits / max(1, len(exp))
    s = 1.0 if (name_ok and fuzzy == 1.0) else (0.5 if name_ok else 0.0)
    return {"valid": valid, "name_ok": name_ok, "fuzzy": fuzzy, "score": s, "parsed": parsed}


def run(model_id, adapter, limit, label):
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    sampler = make_sampler(temp=0.0)
    print(f"\n===== {label}: {model_id} adapter={adapter} =====")
    model, tok = load(model_id, adapter_path=adapter) if adapter else load(model_id)
    rows = []
    for c in CASES[:limit]:
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": c["user"]}],
            tools=c["tools"], tokenize=False, add_generation_prompt=True)
        t0 = time.time()
        out = generate(model, tok, prompt=prompt, max_tokens=100, sampler=sampler, verbose=False)
        dt = time.time() - t0
        s = score(out, c)
        rows.append({"id": c["id"], **s, "latency": round(dt, 2), "output": out[:300]})
        flag = "OK " if s["score"] == 1 else ("HALF" if s["score"] == .5 else "FAIL")
        print(f"[{flag}] {c['id']}: name_ok={s['name_ok']} fuzzy={s['fuzzy']} {dt:.1f}s | {out[:120]!r}")
    del model
    gc.collect()
    return rows


def summary(label, rows):
    n = len(rows)
    avg = lambda k: sum(r[k] for r in rows) / n
    print(f"\n--- {label} --- score={avg('score'):.3f} name={avg('name_ok'):.3f} "
          f"valid={avg('valid'):.3f} lat={avg('latency'):.2f}s")
    return {"model": label, "score": round(avg("score"), 3), "name_acc": round(avg("name_ok"), 3),
            "valid": round(avg("valid"), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--limit", type=int, default=len(CASES))
    ap.add_argument("--out", default="eval_results.json")
    a = ap.parse_args()
    base = run(a.model, None, a.limit, "base")
    sums = [summary("base", base)]
    details = {"base": base}
    if a.adapter:
        tuned = run(a.model, a.adapter, a.limit, "lora")
        sums.append(summary("lora", tuned))
        details["lora"] = tuned
    with open(a.out, "w") as f:
        json.dump({"summary": sums, "details": details}, f, indent=2)
    print(f"\nSaved {a.out}\n{sums}")


if __name__ == "__main__":
    main()
