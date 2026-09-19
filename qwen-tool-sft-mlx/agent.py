"""Mini agent loop on MLX: generate -> execute mock tool -> answer.

Usage:
  python agent.py --prompt "What's the weather in Tokyo?"
  python agent.py --adapter ./adapters --prompt "Compute 128 * 6"
"""
from __future__ import annotations
import argparse, ast, json, re

from eval import TOOLS, parse  # reuse strict parser + schemas

WEATHER = {"tokyo": (16, "Clear"), "london": (11, "Rainy"),
           "san francisco": (18, "Foggy"), "paris": (19, "Partly cloudy")}


def safe_calc(expr: str):
    if not re.fullmatch(r"[0-9+\-*/().\s]+", expr or ""):
        raise ValueError("bad expression")
    tree = ast.parse(expr, mode="eval")
    ok = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
          ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Load,
          ast.UAdd, ast.USub, ast.Pow, ast.FloorDiv)
    for n in ast.walk(tree):
        if not isinstance(n, ok): raise ValueError(f"blocked {type(n).__name__}")
        if isinstance(n, ast.Constant) and not isinstance(n.value, (int, float)):
            raise ValueError("numbers only")
    v = eval(compile(tree, "<e>", "eval"), {"__builtins__": {}}, {})
    return int(v) if isinstance(v, float) and v.is_integer() else v


def execute(name, args):
    if name == "get_weather":
        city = str(args.get("city", "")).strip()
        if not city: return {"error": "missing city"}
        hit = WEATHER.get(city.lower())
        return {"temperature": hit[0], "condition": hit[1], "unit": "celsius"} if hit \
            else {"error": f"no data for {city}", "hint": "try Tokyo, London, Paris, San Francisco"}
    if name == "calculator":
        try: return {"result": safe_calc(str(args.get("expression", "")))}
        except Exception as e: return {"error": str(e)}
    return {"error": f"unknown tool {name}"}


def generate_from(model, tok, messages, tools=TOOLS, max_tokens=160):
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler
    prompt = tok.apply_chat_template(messages, tools=tools,
                                     tokenize=False, add_generation_prompt=True)
    return generate(model, tok, prompt=prompt, max_tokens=max_tokens,
                    sampler=make_sampler(temp=0.0), verbose=False)


def run_agent(model, tok, user_text, rounds=2):
    messages = [{"role": "user", "content": user_text}]
    for i in range(rounds + 1):
        reply = generate_from(model, tok, messages)
        parsed, kind = (None, "TEXT")
        p, valid = parse(reply)
        if valid: parsed, kind = p, "CALL"
        print(f"\n--- round {i} [{kind}] ---\n{reply}")
        if kind != "CALL": return reply
        res = execute(parsed["name"], parsed.get("arguments", {}))
        print(f"--- executed {parsed['name']} -> {res} ---")
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": f"<tool_response>\n{json.dumps(res)}\n</tool_response>"})
    return reply


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--prompt", default="What's the weather in Tokyo?")
    a = ap.parse_args()
    from mlx_lm import load
    model, tok = load(a.model, adapter_path=a.adapter) if a.adapter else load(a.model)
    final = run_agent(model, tok, a.prompt)
    print(f"\nFinal: {final}")


if __name__ == "__main__":
    main()
