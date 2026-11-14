#!/usr/bin/env python3
"""Decode speed of structured output against free text, on the running server.

    python3 tools/structured_bench.py [--url http://127.0.0.1:8888] [--levels 1,4,16] [--runs 3]

The same request three ways, greedy, thinking off, a JSON reply of about 600 tokens:
  free, drafted       no grammar (the MTP head and copy drafts on: the server's normal speed)
  free, draft:false   no grammar, no drafts (what a grammar reply is compared with: it drafts from the prompt only)
  json_schema         the same reply under a JSON schema (response_format)
Each is sent alone, then N at once for each level. Reported: tokens a second of decode (completion tokens over wall time, the
prompt being ~100 tokens so its prefill is a few percent), the median of --runs, and the aggregate over the N at once. The
three modes' replies differ a little in length, which the tokens a second takes out. Python's standard library only.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import statistics
import sys
import time
import urllib.request

ITEMS = 25
PROMPT = (
    f"List {ITEMS} European capital cities as JSON, exactly in this shape and nothing else: "
    '{"cities": [{"name": "...", "country": "...", "population": 123456, "landmarks": ["...", "..."]}]}. '
    "Give each city two landmarks."
)
SCHEMA = {
    "type": "object",
    "properties": {
        "cities": {
            "type": "array",
            "minItems": ITEMS,
            "maxItems": ITEMS,
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "country": {"type": "string"},
                    "population": {"type": "integer"},
                    "landmarks": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2},
                },
                "required": ["name", "country", "population", "landmarks"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["cities"],
    "additionalProperties": False,
}
MODES = [
    ("free, drafted", {}),
    ("free, draft:false", {"draft": False}),
    ("json_schema", {"response_format": {"type": "json_schema", "json_schema": {"name": "cities", "schema": SCHEMA}}}),
]


def post(url: str, body: dict) -> dict:
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())


def request(url: str, model: str, extra: dict) -> tuple[int, float, str]:
    body = {"model": model, "messages": [{"role": "user", "content": PROMPT}], "temperature": 0, "max_tokens": 2500,
            "chat_template_kwargs": {"enable_thinking": False}, **extra}
    t = time.monotonic()
    r = post(url, body)
    wall = time.monotonic() - t
    return r["usage"]["completion_tokens"], wall, (r["choices"][0]["message"].get("content") or "")


def valid_json(text: str) -> bool:
    try:
        v = json.loads(text)
        return isinstance(v, dict) and len(v.get("cities", [])) == ITEMS
    except ValueError:
        return False


def level(url: str, model: str, extra: dict, n: int, runs: int) -> tuple[float, float, int, int]:
    """(tokens a second per request, aggregate tokens a second, mean reply tokens, replies that are the expected JSON)."""
    per, agg, lengths, good = [], [], [], 0
    for _ in range(runs):
        t = time.monotonic()
        with cf.ThreadPoolExecutor(n) as pool:
            done = list(pool.map(lambda _i: request(url, model, extra), range(n)))
        wall = time.monotonic() - t
        toks = [d[0] for d in done]
        per.append(statistics.mean(d[0] / d[1] for d in done))
        agg.append(sum(toks) / wall)
        lengths.append(statistics.mean(toks))
        good += sum(valid_json(d[2]) for d in done)
    return statistics.median(per), statistics.median(agg), round(statistics.median(lengths)), good // runs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://127.0.0.1:8888")
    ap.add_argument("--levels", default="1,4,16", help="requests at once, comma separated")
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()
    levels = [int(x) for x in args.levels.split(",")]
    with urllib.request.urlopen(args.url + "/v1/models") as r:
        model = json.loads(r.read())["data"][0]["id"]
    print(f"{model}: {ITEMS} cities as JSON, greedy, thinking off, median of {args.runs} runs")
    for _, extra in MODES:  # a warm-up each: kernels, caches and the grammar compiler
        request(args.url, model, extra)
    rows: dict[str, dict[int, tuple[float, float, int, int]]] = {}
    for name, extra in MODES:
        rows[name] = {n: level(args.url, model, extra, n, args.runs) for n in levels}
    print(f"\n{'mode':20}" + "".join(f"{f'{n} at once':>26}" for n in levels))
    print(f"{'':20}" + "".join(f"{'per request / total tok/s':>26}" for _ in levels))
    for name, _ in MODES:
        cells = "".join(f"{f'{rows[name][n][0]:.1f} / {rows[name][n][1]:.1f}':>26}" for n in levels)
        print(f"{name:20}{cells}")
    base, plain, grammar = (rows[m[0]] for m in MODES)
    print("\njson_schema against free, drafted (per request):  " + "  ".join(f"{n}: {grammar[n][0] / base[n][0]:.2f}x" for n in levels))
    print("json_schema against free, draft:false:            " + "  ".join(f"{n}: {grammar[n][0] / plain[n][0]:.2f}x" for n in levels))
    bad = [(name, n) for name, _ in MODES[2:] for n in levels if rows[name][n][3] != n]
    print("reply tokens (mean): " + ", ".join(f"{name} {rows[name][levels[0]][2]}" for name, _ in MODES))
    for name, n in bad:
        print(f"WARN {name} at {n}: {rows[name][n][3]} of {n} replies were not the expected JSON")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
