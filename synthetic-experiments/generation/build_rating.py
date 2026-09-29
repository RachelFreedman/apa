"""Collect generated responses, check constraints, and build the blind rating prompt.
Reads ../data/generated_*.json; writes ../prompts/rate_all.txt and ../data/rating_key.json (maps the shuffled
answer ids R01..R78 shown to the raters back to item keys such as Q1_03)."""
import json, os, re, random
from design import DEMO, CALIB, RATE_PROMPT, foundation_block
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = lambda f: os.path.join(ROOT, 'data', f)
items = []   # (item_key, question_text, response_text)
for qid in DEMO:
    gen = json.load(open(D(f'generated_{qid}.json')))
    assert [g['target_id'] for g in gen] == list(range(1, len(DEMO[qid]['targets']) + 1))
    items += [(f"{qid}_{g['target_id']:02d}", DEMO[qid]['question'], g['text']) for g in gen]
cal = json.load(open(D('generated_calibration.json')))
for cid, q in CALIB.items():
    assert len(cal[cid]) == 8
    items += [(f"{cid}_{i+1:02d}", q, t) for i, t in enumerate(cal[cid])]
banned = re.compile(r"\b(care|caring|equal\w*|proportional\w*|loyal\w*|authorit\w*|pur(e|ity|ify|ified))\b", re.I)
wc = [len(t.split()) for _, _, t in items]
print(f"{len(items)} items; words min {min(wc)} max {max(wc)} mean {sum(wc)/len(wc):.1f}")
for k, _, t in items:
    if banned.findall(t): print("  banned-word hit:", k, [x if isinstance(x, str) else x[0] for x in banned.findall(t)])
rng = random.Random(20260928)
order = list(range(len(items))); rng.shuffle(order)
key = {}; lines = []
for n, idx in enumerate(order, 1):
    rid = f"R{n:02d}"; k, q, t = items[idx]; key[rid] = k
    lines.append(f'{rid}. Question: "{q}"\n    Answer: "{t}"')
prompt = RATE_PROMPT.format(foundations=foundation_block(), items="\n\n".join(lines))
open(os.path.join(ROOT, 'prompts', 'rate_all.txt'), 'w').write(prompt)
json.dump(key, open(D('rating_key.json'), 'w'), indent=0)
print(len(prompt.split()), "words in rating prompt")
