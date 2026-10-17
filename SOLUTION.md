# Agent overload: what I changed and how I tested it

## The problem

OpenPoke's interaction agent routes work to execution agents. Agents are never deleted, and on **every turn** the whole roster is pasted into the prompt as bare names:

```xml
<agent name="Email to Alice" />
<agent name="Tokyo trip" />
... (x 300)
```

That breaks down in three ways as the roster grows:

1. **Cost and latency.** At 1,000 agents the list alone is ~13k tokens per turn, before the agent has done anything.
2. **Worse routing.** A name is all the model has to go on. "Email to Alice" doesn't say *which* Alice or *what* it was about, and with hundreds of entries the right one is easy to miss.
3. **Duplicates.** Reuse only happens on an exact string match, so `"Email Alice"` creates a brand new agent next to `"Email to Alice"`, with none of the thread's history. Duplicates make the roster grow even faster.

## The fix

**1. Give each agent a record, not just a name** (`server/services/execution/roster.py`)
Each agent now stores what it was created for, what it was last asked, a snippet of its last result, when it was last used, and how often. Old `roster.json` files (a list of names) are upgraded automatically. Saves are now atomic (the original truncated the file before locking it, so a crash could wipe the roster).

**2. Show a shortlist instead of the whole roster** (`server/services/execution/agent_selector.py`)
Each turn the interaction agent sees at most ~11 agents:
- the **top 8 most relevant** to the current message (plus a little weight on the last few turns, for follow-ups), and
- the **3 most recently used** (a "hot cache" for messages like *"did she reply?"* that name nothing).

Relevance = keyword match (BM25) over name + task + last result, optional semantic match with a small local embedding model (`fastembed`, runs on the laptop, free), and a recency boost that only breaks ties between agents that already match. Each entry shows its purpose and latest result, so the model can tell the two Alices apart.

**3. A way to look past the shortlist** (`search_agents` tool)
Capping the list can't make an agent permanently invisible. The prompt says how many agents are hidden, and the model can call `search_agents("landlord lease")` to search all of them before deciding to create a new one.

**4. Duplicate guard** (`send_message_to_agent`)
Before creating an agent, the name is normalized (case, word order, filler words like "to"/"re") and lightly typo-matched. `"Email Alice"`, `"alice email"` and `"Email to Alcie"` all reuse `"Email to Alice"`. It's deliberately conservative: `"Bob"` vs `"Rob"` and `"Email to Alice"` vs `"Email to Alice re budget"` stay separate, because merging two different threads is worse than one extra agent.

## How I tested it

### Test data
`evals/synthetic_roster.py` builds seeded rosters of 25 / 100 / 300 / 1,000 agents: 22 hand-written "anchor" agents plus generated filler that looks like weeks of use (emails, reminders, receipts, trips). Several anchors are traps: Alice Chen vs Alice Park, Bob vs Rob, Tokyo vs Kyoto, and filler adds more Alices and hotels as the roster grows.

`evals/cases.py` has 40 labelled messages in 6 categories:

| category | example | what it tests |
|---|---|---|
| direct | "Did Alice Chen reply about lunch?" | basic lookup |
| paraphrase | "What did my landlord say about renewing?" | different wording |
| distractor | "Reply to Alice Park about the budget" | confusable agents |
| follow_up | "did she reply yet?" (+ recent conversation) | no keywords at all |
| hard_paraphrase | "When am I heading to Asia?" | zero word overlap, expected to fail keyword matching |
| new_task | "Email Carla about the hackathon" | should create a new agent, not reuse one |

### 1. Unit tests (`tests/`, `python -m pytest -q`)
19 tests: old roster format upgrades, records update on use, the duplicate guard (including the cases it must *not* merge), shortlist size stays bounded at 1,000 agents, the right Alice wins, follow-ups use context, recency breaks ties, `search_agents` finds hidden agents, and the LLM eval harness grades correctly against a scripted fake model.

### 2. Retrieval eval (`python -m evals.retrieval_eval`, no API calls)
Is the correct agent in what the model sees, and what does it cost?

| agents | strategy | correct agent shown | ranked #1 | tokens / turn |
|---:|---|---:|---:|---:|
| 100 | original (all names) | 100% | n/a | 1,169 |
| 100 | naive: 11 most recent | 26% | 3% | 132 |
| 100 | **shortlist** | **89%** | 83% | 464 |
| 1000 | original (all names) | 100% | n/a | 13,096 |
| 1000 | naive: 11 most recent | 3% | 0% | 151 |
| 1000 | **shortlist** | **89%** | 86% | **501** |

(Keyword-only. Run it with `fastembed` installed to add a `shortlist+sem` row.)

What this says:
- The shortlist's prompt cost is **flat** (~500 tokens) no matter how big the roster gets, about **26x smaller** than the original at 1,000 agents.
- Just keeping the most recent agents (the obvious cheap fix) is useless: 3% at 1,000 agents.
- Every keyword-only miss is a `hard_paraphrase` case, which is by design. Those are exactly what embeddings and `search_agents` are for.
- "Original" is 100% here only because it shows *everything*. Whether the model then *picks* the right one is what the next eval measures.

### 3. LLM routing eval (`python -m evals.llm_eval`, uses the free OpenRouter tier)
Real system prompt, real tool schemas, 300-agent roster, same cases. Graded as `correct`, `duplicate` (created a new agent when one existed), `wrong_agent`, or `no_route`.

RESULTS_PLACEHOLDER

## Limitations and what I'd do next

- **My test cases were written by the same person who wrote the selector.** That's a real bias. Next step: sample real (anonymized) user messages and label them, or have a second person write cases blind.
- **Keyword matching can't do pure paraphrase.** Embeddings help, but the real backstop is `search_agents`, which depends on the model deciding to call it.
- **Execution agents also overload.** Each execution agent loads its *entire* history into its own prompt (`conversation_limit` defaults to `None`). Long-lived agents like "Nightly email summary" grow forever. Same idea applies: summarize old history and keep the recent part.
- **No cleanup.** The prompt is now bounded, but the roster file still grows. Agents untouched for months could be archived (still searchable, never shown by default).
- **Agent descriptions are raw instruction snippets.** A one-line LLM-written summary per agent (on a cheap model, only when the agent changes) would rank better than the first 200 characters of its instructions.

## Running it

```bash
pip install -r server/requirements.txt -r requirements-dev.txt
pip install fastembed               # optional: semantic matching
python -m pytest -q                 # unit tests
python -m evals.retrieval_eval      # retrieval eval, no API key needed
python -m evals.llm_eval            # LLM eval, needs OPENROUTER_API_KEY in .env
```

Tuning knobs (env vars): `OPENPOKE_ROUTER_TOP_K` (default 8), `OPENPOKE_ROUTER_RECENT` (default 3), `OPENPOKE_SEMANTIC_ROUTING=0` to turn off embeddings.
