"""Extended held-out benchmark: base vs LoRA, eval-only (no training).

Unseen cities / numbers on purpose — train pool never saw:
  Reykjavik, Lima, Zurich, Athens, Helsinki, Singapore, + new exprs.
Same official template as eval.py:
  tokenizer.apply_chat_template(msgs, tools=..., add_generation_prompt=True)
Greedy (temp=0.0) for determinism.

Usage:
  python3 eval_benchmark.py                 # base only
  python3 eval_benchmark.py --adapter ./adapters   # base vs LoRA
"""
from __future__ import annotations
import argparse, gc, json, time

from eval import TOOLS, parse  # single source of truth for schema/parser

# cat: weather | calc | pick | no_tool | clarify | tricky
CASES = [
    # --- weather, unseen cities, new phrasings ---
    {"id": "w_unseen_reykjavik", "cat": "weather", "user": "What's the weather like in Reykjavik right now?",
     "tools": [TOOLS[0]], "name": "get_weather", "args": {"city": "Reykjavik"}},
    {"id": "w_unseen_lima_short", "cat": "weather", "user": "Weather in Lima?",
     "tools": TOOLS, "name": "get_weather", "args": {"city": "Lima"}},
    {"id": "w_unseen_fahrenheit", "cat": "weather", "user": "Temperature in Zurich in fahrenheit?",
     "tools": [TOOLS[0]], "name": "get_weather", "args": {"city": "Zurich", "unit": "fahrenheit"}},
    {"id": "w_unseen_athens", "cat": "weather", "user": "And how about Athens?",
     "tools": TOOLS, "name": "get_weather", "args": {"city": "Athens"}},

    # --- calc, new numbers, word->symbol, parens, decimals ---
    {"id": "c_simple", "cat": "calc", "user": "Compute 47 * 13",
     "tools": [TOOLS[1]], "name": "calculator", "args": {"expression": "47 * 13"}},
    {"id": "c_word_div", "cat": "calc", "user": "What is 144 divided by 12?",
     "tools": TOOLS, "name": "calculator", "args": {"expression": "144 / 12"}},
    {"id": "c_word_mult", "cat": "calc", "user": "Compute 45 multiplied by 12",
     "tools": TOOLS, "name": "calculator", "args": {"expression": "45 * 12"}},
    {"id": "c_paren_new", "cat": "calc", "user": "What is (23 + 17) * 4?",
     "tools": TOOLS, "name": "calculator", "args": {"expression": "(23 + 17) * 4"}},
    {"id": "c_paren_div", "cat": "calc", "user": "Evaluate (100 - 37) / 9",
     "tools": [TOOLS[1]], "name": "calculator", "args": {"expression": "(100 - 37) / 9"}},
    {"id": "c_decimal", "cat": "calc", "user": "Calculate 7.5 * 4",
     "tools": [TOOLS[1]], "name": "calculator", "args": {"expression": "7.5 * 4"}},

    # --- tool choice with BOTH tools offered ---
    {"id": "p_calc", "cat": "pick", "user": "What is 15 plus 27?",
     "tools": TOOLS, "name": "calculator", "args": {"expression": "15 + 27"}},
    {"id": "p_weather_unseen", "cat": "pick", "user": "How's the weather in Singapore today?",
     "tools": TOOLS, "name": "get_weather", "args": {"city": "Singapore"}},

    # --- no-tool: fresh trivia / chit-chat NOT in train ---
    {"id": "n_trivia_orwell", "cat": "no_tool", "user": "Who wrote 1984?",
     "tools": TOOLS, "name": None, "args": None},
    {"id": "n_trivia_italy", "cat": "no_tool", "user": "What is the capital of Italy?",
     "tools": [TOOLS[0]], "name": None, "args": None},
    {"id": "n_chat_evening", "cat": "no_tool", "user": "Good evening! What's up?",
     "tools": TOOLS, "name": None, "args": None},
    {"id": "n_thanks", "cat": "no_tool", "user": "Thanks for your help!",
     "tools": TOOLS, "name": None, "args": None},

    # --- clarify: missing arg -> ask, don't hallucinate ---
    {"id": "q_weather_bare", "cat": "clarify", "user": "Check the weather please.",
     "tools": [TOOLS[0]], "name": None, "args": None, "must_ask": "city"},
    {"id": "q_temp", "cat": "clarify", "user": "Tell me the temperature.",
     "tools": [TOOLS[0]], "name": None, "args": None, "must_ask": "city"},
    {"id": "q_math", "cat": "clarify", "user": "Can you calculate something for me?",
     "tools": TOOLS, "name": None, "args": None, "must_ask": "expr"},
    {"id": "q_raining_nocity", "cat": "clarify", "user": "Is it raining right now?",
     "tools": [TOOLS[0]], "name": None, "args": None, "must_ask": "city"},

    # --- tricky: same words, different intent ---
    {"id": "t_raining_paris", "cat": "tricky", "user": "Is it raining in Paris?",
     "tools": [TOOLS[0]], "name": "get_weather", "args": {"city": "Paris"}},
    {"id": "t_helsinki_nounit", "cat": "tricky", "user": "Weather in Helsinki?",
     "tools": [TOOLS[0]], "name": "get_weather", "args": {"city": "Helsinki"},
     "no_unit_hallucination": True},
]


def norm_expr(s):
    # "144 / 12" == "144/12"; case/space insensitive. Word forms stay distinct (intended).
    return s.strip().lower().replace(" ", "") if isinstance(s, str) else s


def norm(v):
    return v.strip().lower() if isinstance(v, str) else v


def score(text, case):
    parsed, valid = parse(text)
    if case["name"] is None:
        if "<tool_call" in text or parsed is not None:
            return {"valid": False, "name_ok": False, "fuzzy": 0.0, "score": 0.0,
                    "parsed": parsed, "note": "should_not_call"}
        t = text.lower()
        if case.get("must_ask") == "city":
            ok = ("?" in text) and any(w in t for w in ("which city", "which", "where"))
            # guard: chit-chat must not trigger city-ask path (handled in no_tool below),
            # here we REQUIRE the ask.
            return {"valid": valid, "name_ok": ok, "fuzzy": 1.0 if ok else 0.0,
                    "score": 1.0 if ok else 0.0, "parsed": None, "note": "clarify_city"}
        if case.get("must_ask") == "expr":
            ok = ("?" in text) and any(w in t for w in ("express", "comput", "calcul", "evaluat", "solve", "what should", "what to"))
            return {"valid": valid, "name_ok": ok, "fuzzy": 1.0 if ok else 0.0,
                    "score": 1.0 if ok else 0.0, "parsed": None, "note": "clarify_expr"}
        # plain no_tool: must NOT ask for a city (old LoRA said "Which city?" to "hey")
        if "which city" in t or "which city should" in t:
            return {"valid": valid, "name_ok": False, "fuzzy": 0.0, "score": 0.0,
                    "parsed": None, "note": "spurious_city_ask"}
        return {"valid": valid, "name_ok": True, "fuzzy": 1.0, "score": 1.0,
                "parsed": None, "note": "no_call_ok"}
    if parsed is None:
        return {"valid": False, "name_ok": False, "fuzzy": 0.0, "score": 0.0,
                "parsed": None, "note": "no_parse"}
    name_ok = parsed.get("name") == case["name"]
    args = parsed.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {}
    exp = case["args"] or {}
    hits = 0
    for k, ev in exp.items():
        if k not in args:
            continue
        av = args[k]
        if k == "expression":
            if norm_expr(av) == norm_expr(ev):
                hits += 1
        else:
            if norm(av) == norm(ev):
                hits += 1
            elif isinstance(ev, str) and isinstance(av, str) and norm(ev) in norm(av):
                hits += 0.5
    fuzzy = hits / max(1, len(exp))
    s = 1.0 if (name_ok and fuzzy == 1.0) else (0.5 if name_ok else 0.0)
    note = "ok"
    # flag unit hallucination: user didn't ask for a unit but model emitted one
    if case.get("no_unit_hallucination") and isinstance(args, dict) and "unit" in args:
        note = "unit_hallucinated"
    if case["id"] == "w_unseen_fahrenheit" and isinstance(args, dict) and args.get("unit") != "fahrenheit":
        note = "unit_missing"
    return {"valid": valid, "name_ok": name_ok, "fuzzy": fuzzy, "score": s,
            "parsed": parsed, "note": note}


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
        rows.append({"id": c["id"], "cat": c["cat"], **s, "latency": round(dt, 2), "output": out[:300]})
        flag = "OK " if s["score"] == 1 else ("HALF" if s["score"] == .5 else "FAIL")
        print(f"[{flag}] {c['id']} ({c['cat']}): name_ok={s['name_ok']} fuzzy={s['fuzzy']} note={s['note']} {dt:.1f}s | {out[:120]!r}")
    del model
    gc.collect()
    return rows


def summary(label, rows):
    n = len(rows)
    avg = lambda k: sum(r[k] for r in rows) / n
    print(f"\n--- {label} --- score={avg('score'):.3f} name={avg('name_ok'):.3f} "
          f"valid={avg('valid'):.3f} lat={avg('latency'):.2f}s")
    cats = {}
    for cat in sorted(set(r["cat"] for r in rows)):
        sub = [r for r in rows if r["cat"] == cat]
        cats[cat] = round(sum(r["score"] for r in sub) / len(sub), 3)
        print(f"    {cat:8s} {cats[cat]:.3f} ({len(sub)} cases)")
    return {"model": label, "score": round(avg("score"), 3), "name_acc": round(avg("name_ok"), 3),
            "valid": round(avg("valid"), 3), "by_cat": cats}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--limit", type=int, default=len(CASES))
    ap.add_argument("--out", default="eval_benchmark_results.json")
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
    print(f"\nSaved {a.out}\n{json.dumps(sums, indent=2)}")


if __name__ == "__main__":
    main()
