# Millok

**Turn an agent's own mistakes into fine-tuning data — automatically, with zero manual labeling.**

> A correction is training data you already paid for.

Any agent that uses tools (calls functions, runs commands, controls things) fails sometimes — and sometimes it succeeds while admitting it wasn't sure. Both moments are labeled training data, for free, if you're logging them:

```
failed attempt  ─┐
                  ├─ same intent, close together, same session ─→  (rejected, chosen) pair
succeeded later ─┘

succeeded, but flagged as uncertain ─→ reinforce it — no wrong attempt needed
```

If the retry succeeds, or the hedge turns out to have been right, **you now have a labeled example** — "here's what went wrong, here's what right looks like" or "here's what you weren't sure about, but got right" — and nobody had to sit down and write it. The agent's own operation already did the labeling.

Millok finds those moments in your logs and turns them into a dataset.

## Why this exists

Generic fine-tuning datasets don't know that *your* model, in *your* application, keeps confusing two specific tools, or keeps getting one argument format wrong. That knowledge only exists in your own operational history — and it's usually thrown away the moment the retry succeeds and the conversation moves on. Any tool-using agent that logs its attempts already has this data sitting around, unused.

## What it actually does (and doesn't)

Millok mines two shapes of signal from a sequence of tool-use attempts, and lists a third thing that needs no training at all:

- **`extract_pairs`** — a failed attempt followed by a similar-intent success nearby → `(rejected, chosen)`.
- **`extract_confirmations`** — a success the agent itself flagged as uncertain → reinforce it, no wrong attempt needed.
- **`extract_gaps`** — an intent that was never resolved. No dataset comes out of this one; it's a straight list of "still doesn't work".

That's the whole scope.

- No embeddings, no external model, no GPU, no network access. The whole matching step is `difflib` from the Python standard library.
- It does not train anything. Output is standard DPO/SFT-format JSONL — plug it into [TRL](https://github.com/huggingface/trl), [unsloth](https://github.com/unslothai/unsloth), [axolotl](https://github.com/OpenAccess-AI-Collective/axolotl), or whatever you already use.
- It does not judge whether a "success" was actually correct, or whether "uncertain" was flagged honestly — that's your system's job to log accurately. Millok just notices the pattern.
- It works best for **tool-use / structured-output agents**, where "success" and "failure" are well-defined. It has nothing useful to say about open-ended chit-chat.

## Quickstart

```bash
git clone <this repo>
cd Millok
python3 -m millok.cli demo
```

That runs Millok against a built-in synthetic log and prints what it found — no setup, no API keys, no downloads. Real output from this repo:

```
synthetic log      : 27 turns across 18 sessions
failed attempts    : 9
pairs mined        : 8
confirmed-uncertain: 1
unresolved gaps    : 1

sample pair:
  intent   : search for weather sites
  rejected : search_web(engine='duckduckgo', q=None)
  reason   : missing required argument 'q'
  chosen   : search_web(engine='duckduckgo', q='weather')
  weight   : 5  (this exact failure reason recurred 5 times)

sample confirmation (succeeded, but the agent had hedged):
  intent   : turn off the desk lamp
  attempt  : smart_plug(id='lamp_2', on=False)
  reason   : two devices are named similarly ('lamp_1', 'lamp_2') - picked the one used most recently, but wasn't certain which one was meant

gap:
  intent   : send a message to the whole team at once
  reason   : unknown recipient group 'team' - no such contact list exists
```

Three things worth noticing in that output:

- **`weight: 5`** — this failure mode showed up in five separate sessions. A one-off typo and a systematic blind spot both produce a pair; the weight is what tells a training script which one actually matters. (See `_weight_by_recurrence` in `millok/mining.py`.)
- **The confirmation** — nothing was ever wrong here. The agent picked correctly, said so uncertainly, and got no correction — which is itself the signal: reinforce this, stop hedging on it.
- **The gap** — an intent that was *never* resolved. Even with no training step at all, this is immediately useful: it's the shortest path to "what does this agent still not understand."

## Using it on your own logs

Log format is one JSON object per line — this is the only integration point:

```json
{"session": "chat-42", "intent": "open the budget file", "attempt": "open_flie(name='budget')", "success": false, "reason": "unknown tool 'open_flie'"}
{"session": "chat-42", "intent": "open the budget file", "attempt": "open_file(name='budget')", "success": true, "reason": ""}
```

`reason` means two different things depending on `success` — on a failure, why it broke; on a success, why the agent hedged (leave it empty for a clean, confident success):

```json
{"session": "chat-77", "intent": "turn off the desk lamp", "attempt": "smart_plug(id='lamp_2', on=False)", "success": true, "reason": "two devices named similarly, picked the closer match"}
```

Then:

```bash
python3 -m millok.cli mine your_log.jsonl --format dpo -o dataset.jsonl
python3 -m millok.cli confirmed your_log.jsonl -o reinforce.jsonl
python3 -m millok.cli gaps your_log.jsonl
```

Or as a library:

```python
from millok import read_turns, extract_pairs, extract_confirmations, extract_gaps, to_dpo_jsonl

turns = read_turns("your_log.jsonl")
pairs = extract_pairs(turns, window=5, threshold=0.6)
to_dpo_jsonl(pairs, "dataset.jsonl")

for hedge in extract_confirmations(turns):
    print("stop hedging on:", hedge.intent)

for gap in extract_gaps(turns):
    print(gap.intent, "->", gap.reason)
```

`window` controls how many turns a failure waits for its fix before being considered too far away to trust as a clean signal — a correction that shows up fifty turns later, in a different context, is coincidence, not supervision. `threshold` controls how similar two intents must be to count as "the same request, retried" (0–1, via `difflib.SequenceMatcher`).

## Feeding the output into a real trainer

Millok's job ends at the JSONL file. Example with TRL:

```python
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig

dataset = load_dataset("json", data_files="dataset.jsonl")["train"]
trainer = DPOTrainer(model=..., args=DPOConfig(...), train_dataset=dataset)
trainer.train()
```

LoRA-style fine-tuning on the resulting dataset is feasible on a single consumer GPU (8GB+ VRAM) for 7–8B models — this isn't a large-scale-compute idea.

## Running the tests

```bash
python3 tests/test_mining.py -v
```

25 checks, no test framework dependency, plain asserts — covering session isolation (a fix in one conversation must never "fix" another), window expiry, many-failures-to-one-success, recurrence weighting, threshold sensitivity, and that confirmations never get confused with regular pairs.

## Design notes

The core algorithm is deliberately small (`millok/mining.py`, under 100 lines) and has zero dependencies outside the standard library. That's not an accident or a limitation — matching "is this the same intent, retried" turns out not to need anything fancier than a fuzzy string comparison, because retried intents are almost always near-identical text (a misheard command repeated, a request rephrased slightly). Reaching for embeddings or a classifier here would be solving a harder problem than the one that actually exists.

## License

MIT — see `LICENSE`.
