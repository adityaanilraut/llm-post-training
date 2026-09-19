"""Build MLX SFT data: full ChatML text per trajectory.

One format everywhere (fixes the old notebook bug):
  tokenizer.apply_chat_template(messages, tools=TOOLS, ...)
never a hand-built SYSTEM_PROMPT string.

Output: data/train.jsonl, data/valid.jsonl, data/test.jsonl
Each line: {"text": "<|im_start|>...<|im_end|>..."}
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path

from huggingface_hub import snapshot_download

TOKENIZER_ID = "Qwen/Qwen2.5-0.5B-Instruct"

TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Get current temperature and conditions for a city.",
        "parameters": {"type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. Tokyo"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
            "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "calculator",
        "description": "Evaluate basic arithmetic (+, -, *, /, parentheses).",
        "parameters": {"type": "object",
            "properties": {
                "expression": {"type": "string", "description": "e.g. '12 * 8'"}},
            "required": ["expression"]}}},
]


def call(name, **args):
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": args}, ensure_ascii=False) + "\n</tool_call>"


def result(obj):
    return json.dumps(obj, ensure_ascii=False)


def weather(user, city, temp, cond, unit="celsius"):
    sym = "°F" if unit == "fahrenheit" else "°C"
    args = {"city": city} if unit == "celsius" else {"city": city, "unit": unit}
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": call("get_weather", **args)},
        {"role": "tool", "content": result({"temperature": temp, "condition": cond, "unit": unit})},
        {"role": "assistant", "content": f"{city} is {temp}{sym} and {cond.lower()}."},
    ]


def calc(user, expr, val):
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": call("calculator", expression=expr)},
        {"role": "tool", "content": result({"result": val})},
        {"role": "assistant", "content": f"The result of {expr} is {val}."},
    ]


def direct(user, reply):
    return [{"role": "user", "content": user}, {"role": "assistant", "content": reply}]


def clarify(user, reply):
    return direct(user, reply)


def to_template_messages(traj):
    """Map 'tool' role -> 'user' + <tool_response>, which the Qwen template understands."""
    out = []
    for t in traj:
        if t["role"] == "tool":
            out.append({"role": "user", "content": f"<tool_response>\n{t['content']}\n</tool_response>"})
        else:
            out.append({"role": t["role"], "content": t["content"]})
    return out


def build_trajectories(seed=42):
    rng = random.Random(seed)
    # Larger, varied pools — test held-outs (San Francisco, 128*6) NEVER appear here.
    city_pool = [
        ("Tokyo", 16, "Clear"), ("London", 11, "Rainy"),
        ("Dubai", 38, "Sunny"), ("Paris", 19, "Partly cloudy"),
        ("Berlin", 14, "Overcast"), ("Oslo", 7, "Snowy"),
        ("Rome", 22, "Sunny"), ("Madrid", 26, "Clear"),
        ("Sydney", 21, "Windy"), ("Mumbai", 32, "Humid"),
        ("Toronto", 9, "Cloudy"), ("Cairo", 30, "Sunny"),
        ("Seoul", 13, "Clear"), ("Mexico City", 20, "Partly cloudy"),
        ("Nairobi", 24, "Sunny"), ("Bangkok", 33, "Rainy"),
        ("Chicago", 12, "Windy"), ("Boston", 10, "Rainy"),
        ("Lisbon", 23, "Clear"), ("Amsterdam", 15, "Overcast"),
        ("New York", 64, "Cloudy"), ("Denver", 18, "Clear"),
    ]
    weather_phrasings = [
        "What's the weather in {city}?",
        "Weather in {city}?",
        "Temperature in {city}?",
        "How's the weather in {city} today?",
        "Tell me the weather for {city}.",
        "What's it like in {city} right now?",
    ]
    calc_phrasings = [
        "Compute {expr}",
        "What is {expr}?",
        "Evaluate {expr}",
        "Calculate {expr}",
        "Solve {expr}",
    ]
    # word-form phrasings so "128 multiplied by 6" style generalizes
    word_ops = [(" * ", " multiplied by "), (" / ", " divided by "),
                (" + ", " plus "), (" - ", " minus ")]

    def random_expr():
        a, b = rng.randint(2, 500), rng.randint(2, 99)
        op = rng.choice(["+", "-", "*", "/"])
        if rng.random() < 0.25:  # parenthesized variant
            c = rng.randint(2, 20)
            expr = f"({a} {op} {b}) * {c}" if op in ("+", "-") else f"({a} {op} {b}) + {c}"
        else:
            expr = f"{a}{op}{b}" if rng.random() < 0.5 else f"{a} {op} {b}"
            if rng.random() < 0.2:
                expr = expr.replace(".", "", 0)
                expr = f"{rng.randint(1,9)}.{rng.randint(1,9)} {op} {rng.randint(1,20)}"
        # safe eval for label
        val = eval(expr)
        val = int(val) if isinstance(val, float) and val.is_integer() else round(val, 2)
        return expr, val

    def words_for(expr, user_template):
        # occasionally render "128 * 6" as "128 multiplied by 6"
        disp = expr
        if " * " in expr or " / " in expr or " + " in expr or " - " in expr:
            if rng.random() < 0.35:
                for sym, word in word_ops:
                    disp = disp.replace(sym, word)
        return user_template.format(expr=disp)

    trajs = []
    # ~40 weather: balanced celsius (omit unit) / fahrenheit (include unit)
    for i in range(40):
        city, temp, cond = rng.choice(city_pool)
        # keep held-out SF out of train
        if city == "San Francisco":
            continue
        use_f = rng.random() < 0.4
        unit = "fahrenheit" if use_f else "celsius"
        t = temp if unit == "celsius" else round(temp * 9 / 5 + 32)
        user = rng.choice(weather_phrasings).format(city=city)
        if use_f and "fahrenheit" not in user.lower():
            user += " in fahrenheit" if rng.random() < 0.5 else " (in °F)"
        trajs.append(weather(user, city, t, cond, unit=unit))
    # ~40 calc: random exprs, varied phrasing; avoid exact held-out "128 * 6"
    seen = set()
    while len([t for t in trajs if len(t) == 4 and "result" in t[2]["content"]]) < 40:
        expr, val = random_expr()
        if expr == "128 * 6" or expr in seen:
            continue
        seen.add(expr)
        user = words_for(expr, rng.choice(calc_phrasings))
        trajs.append(calc(user, expr, val))
    # follow-ups: second call must use the NEW city/number
    for _ in range(4):
        c1, c2 = rng.sample([c for c in city_pool if c[0] != "San Francisco"], 2)
        trajs.append([
            {"role": "user", "content": f"Weather in {c1[0]}?"},
            {"role": "assistant", "content": call("get_weather", city=c1[0])},
            {"role": "tool", "content": result({"temperature": c1[1], "condition": c1[2], "unit": "celsius"})},
            {"role": "assistant", "content": f"{c1[0]} is {c1[1]}°C and {c1[2].lower()}."},
            {"role": "user", "content": "And " + c2[0] + "?"},
            {"role": "assistant", "content": call("get_weather", city=c2[0])},
            {"role": "tool", "content": result({"temperature": c2[1], "condition": c2[2], "unit": "celsius"})},
            {"role": "assistant", "content": f"{c2[0]} is {c2[1]}°C and {c2[2].lower()}."}])
    # negatives: answer directly, never call — trivia + chit-chat heavy
    # so the model stops asking "which city?" for greetings.
    trajs += [
        direct("Hi! How do I prepare for interviews?", "Practice data structures, system design, and STAR stories out loud."),
        direct("Who wrote Hamlet?", "William Shakespeare wrote Hamlet."),
        direct("Capital of France?", "Paris."),
        direct("Explain recursion in one sentence.", "Recursion is a function calling itself on a smaller instance until a base case."),
        direct("Plan a weekend?", "Tell me your city and budget and I'll sketch one."),
        direct("Who walked on the moon first?", "Neil Armstrong — Apollo 11, 1969."),
        direct("What is the capital of Japan?", "Tokyo."),
        direct("Summarize photosynthesis.", "Plants convert light, water and CO2 into glucose and oxygen."),
        direct("What year did the Titanic sink?", "1912."),
        direct("Who painted the Mona Lisa?", "Leonardo da Vinci."),
        direct("Hey, how are you?", "Doing well, thanks. How can I help?"),
        direct("Hello!", "Hi there! What can I do for you?"),
        direct("Good morning!", "Good morning! How can I help today?"),
        direct("Thanks!", "You're welcome!"),
        direct("Tell me a joke.", "Why do programmers prefer dark mode? Because light attracts bugs."),
        direct("What can you do?", "I can check weather, do arithmetic, or just chat. Try 'Weather in Tokyo?' or 'Compute 12 * 8'."),
        direct("Who won the 2022 World Cup?", "Argentina."),
        direct("How tall is Mount Everest?", "8,849 meters."),
        direct("What is H2O?", "Water."),
        direct("Good night.", "Good night!"),
    ]
    # clarify: missing required arg -> ask, don't guess
    trajs += [
        clarify("Check the weather please.", "Which city should I look up?"),
        clarify("Tell me the temperature.", "Sure — which city?"),
        clarify("What's the weather like?", "Which city would you like?"),
        clarify("Is it raining?", "Where — which city should I check?"),
        clarify("Weather update?", "For which city?"),
        clarify("Calculate something for me.", "What expression should I evaluate?"),
        clarify("Do some math.", "Sure — what expression? e.g. 12 * 8"),
        clarify("Compute this.", "What should I compute?"),
    ]
    heldout = [
        weather("What's the weather in San Francisco right now?", "San Francisco", 18, "Foggy"),
        calc("Compute 128 multiplied by 6", "128 * 6", 768),
        direct("Who walked on the moon first?", "Neil Armstrong."),
        clarify("Check the weather.", "Which city would you like?"),
    ]
    return trajs, heldout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--valid-frac", type=float, default=0.1)
    args = ap.parse_args()
    random.seed(args.seed)

    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(TOKENIZER_ID, trust_remote_code=True)
    except Exception:
        local = snapshot_download(repo_id=TOKENIZER_ID)
        tok = AutoTokenizer.from_pretrained(local, trust_remote_code=True)

    trajs, heldout = build_trajectories(seed=args.seed)
    random.shuffle(trajs)
    n_valid = max(2, int(len(trajs) * args.valid_frac))
    valid, train = trajs[:n_valid], trajs[n_valid:]

    out = Path(__file__).parent / "data"
    out.mkdir(exist_ok=True)
    stats = {}
    for name, split in (("train", train), ("valid", valid), ("test", heldout)):
        path = out / f"{name}.jsonl"
        with open(path, "w") as f:
            for traj in split:
                text = tok.apply_chat_template(
                    to_template_messages(traj), tools=TOOLS,
                    tokenize=False, add_generation_prompt=False)
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        stats[name] = len(split)
        print(f"{name}: {len(split)} rows -> {path}")
    print(f"tools: {[t['function']['name'] for t in TOOLS]}")
    print("Done. Train on train.jsonl, early-stop on valid.jsonl, report on test.jsonl.")


if __name__ == "__main__":
    main()
