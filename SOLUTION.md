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

## How I thought about it

Agent overload is really a context-budget problem: anything that grows with usage (agents, history, memories) and gets pasted into a prompt on every call will eventually cost too much and bury the useful part. That happens here at two levels: the interaction agent's roster, and each execution agent's own history. So the question I kept asking was **"what does the model need to see on this turn?"**, not "how do we have fewer agents?"

Options I considered:

- **Delete or expire old agents.** Simplest, but it throws away thread history, and the next "did Alice ever reply?" would start from nothing. I went with archiving (hidden by default, still searchable) instead.
- **Merge similar agents into one.** Saves space, but a wrong merge mixes two people's threads, which is worse than one extra agent. I only merge obvious name duplicates, and conservatively.
- **Show only the most recent agents.** Cheap, but my eval shows it finds the right agent 3% of the time at 1,000 agents.
- **Ask a cheap LLM to pick the relevant agents first.** Probably the most accurate, but it adds a model call and latency to every message. Keyword + optional embedding ranking is free and fast, and the model can still search if the shortlist misses.

The approach I took: rank by relevance, show a small shortlist, keep everything reachable through search, and measure whether the right agent actually gets used.

## The fix

**1. Give each agent a record, not just a name** (`server/services/execution/roster.py`)
Each agent now stores what it was created for, what it was last asked, a snippet of its last result, when it was last used, and how often. Old `roster.json` files (a list of names) are upgraded automatically. Saves are now atomic (the original truncated the file before locking it, so a crash could wipe the roster).

**2. Show a shortlist instead of the whole roster** (`server/services/execution/agent_selector.py`)
Each turn the interaction agent sees at most ~11 agents:
- the **top 8 most relevant** to the current message (plus a little weight on the last few turns, for follow-ups), and
- the **3 most recently used** (a "hot cache" for messages like *"did she reply?"* that name nothing).

Relevance = keyword match (BM25) over name + task + last result, optional semantic match with a small local embedding model (`fastembed`, runs on the laptop, free), and a recency boost that only breaks ties between agents that already match. Each entry shows what the agent was created for and what it was last asked, so the model can tell the two Alices apart. The agent's last *result* is used for ranking but deliberately not shown (see the LLM eval below for why).

**3. A way to look past the shortlist** (`search_agents` tool)
Capping the list can't make an agent permanently invisible. The prompt says how many agents are hidden, and the model can call `search_agents("landlord lease")` to search all of them before deciding to create a new one.

**4. Duplicate guard** (`send_message_to_agent`)
Before creating an agent, the name is normalized (case, word order, filler words like "to"/"re") and lightly typo-matched. `"Email Alice"`, `"alice email"` and `"Email to Alcie"` all reuse `"Email to Alice"`. It's deliberately conservative: `"Bob"` vs `"Rob"` and `"Email to Alice"` vs `"Email to Alice re budget"` stay separate, because merging two different threads is worse than one extra agent.

## Other gaps I found and fixed

**5. Execution agents had the same overload problem, one level down** (`server/agents/execution_agent/agent.py`)
Every execution agent pasted its *entire* log into its own prompt on every call (`conversation_limit` defaulted to `None`, so the existing cut-off was never used). A long-lived agent like "Nightly email summary" grows forever, and most of that bulk is old tool output (full email search results). Now each agent sees its **last 6 requests in full**, and older requests are **condensed to one line each** ("asked: … -> replied: …", tool output dropped), with only the last 20 of those kept and the rest counted. No LLM call is needed to condense, so it's free and deterministic. With ~500 characters of tool output per request:

| requests in history | before (tokens) | after (tokens) |
|---:|---:|---:|
| 10 | ~2,100 | ~1,400 |
| 100 | ~21,000 | ~1,800 |
| 365 (a year of nightly runs) | ~77,000 | ~1,800 |

`OPENPOKE_EXEC_HISTORY_TURNS` sets how many requests stay in full (`0` = the old behaviour).

**6. Stale agents stop taking shortlist slots** (`agent_selector.py`)
Agents unused for 60+ days count as **archived**. They aren't deleted or moved, and `search_agents` still finds them. They just need a **strong match** to make the shortlist, so a weak keyword overlap with something from months ago doesn't push out a current agent. Archived is worked out from `last_used_at`, so using an agent again un-archives it automatically and nothing extra is stored. The retrieval eval is unchanged with this on (still 89% at 1,000 agents, including old anchors like "GitHub security alert" at 90 days), so it trims without hiding the agents people actually ask about. `OPENPOKE_ROUTER_ARCHIVE_DAYS` sets the cutoff (`0` = off).

**7. `search_agents` was leaking old results too.** After hiding result snippets from the shortlist (see the LLM eval), I noticed the search tool still returned each agent's last result, which could cause the same "answer from stale text" failure. It now returns what the agent was last *asked* instead, plus an `archived` flag.

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
30 tests. The original 20: old roster format upgrades, records update on use, the duplicate guard (including the cases it must *not* merge), shortlist size stays bounded at 1,000 agents, the right Alice wins, follow-ups use context, recency breaks ties, `search_agents` finds hidden agents, the LLM eval harness grades correctly against a scripted fake model, and a regression test that result snippets are never shown to the model. 10 more for the extra gaps (`tests/test_history_and_archive.py`): short histories are unchanged, old requests are condensed while recent ones stay in full, history size stays bounded from 10 to 1,000 requests, condensed lines drop tool output, archiving is derived from last use (and can be turned off), an old agent still wins on a strong match, weak old matches stop taking slots, the prompt mentions archived agents, and `search_agents` never returns old results.

### 2. Retrieval eval (`python -m evals.retrieval_eval`, no API calls)
Is the correct agent in what the model sees, and what does it cost?

| agents | strategy | correct agent shown | ranked #1 | tokens / turn |
|---:|---|---:|---:|---:|
| 100 | original (all names) | 100% | n/a | 1,169 |
| 100 | naive: 11 most recent | 26% | 3% | 132 |
| 100 | **shortlist** | **89%** | 83% | ~390 |
| 1000 | original (all names) | 100% | n/a | 13,096 |
| 1000 | naive: 11 most recent | 3% | 0% | 151 |
| 1000 | **shortlist** | **89%** | 86% | **~390** |

(Keyword-only. Run it with `fastembed` installed to add a `shortlist+sem` row.)

What this says:
- The shortlist's prompt cost is **flat** (~400 tokens) no matter how big the roster gets, about **30x smaller** than the original at 1,000 agents.
- Just keeping the most recent agents (the obvious cheap fix) is useless: 3% at 1,000 agents.
- Every keyword-only miss is a `hard_paraphrase` case, which is by design. Those are exactly what embeddings and `search_agents` are for.
- "Original" is 100% here only because it shows *everything*. Whether the model then *picks* the right one is what the next eval measures.

### 3. LLM routing eval (`python -m evals.llm_eval`, uses the free OpenRouter tier)
Real system prompt, real tool schemas, 300-agent roster, same cases. Graded as `correct`, `duplicate` (created a new agent when one existed), `wrong_agent`, or `no_route`.

Model: `nvidia/nemotron-3-super-120b-a12b:free`. Small runs, because the free tier allows 50 requests a day, so treat these as directional, not statistically solid.

**Run 1: the shortlist caused a regression.** On 10 cases, the original got 10 of 12 right and the shortlist only 6 of 10. All 4 shortlist failures were the same kind: "Did Alice reply?", "When's my Stripe onsite?", "When's my flight?" and a follow-up. The model never routed them to an agent. The retrieval eval had said these were fine (the right agent *was* shown), so this only showed up end to end.

The cause: I had put each agent's last result in its description ("Alice replied: Thursday works") to help the model choose. For status questions, the model just answered from that snippet instead of asking the agent to check, and the snippet can be days old.

**Run 2: a prompt fix did not work.** I labelled snippets "last known (2d ago, may be outdated)" and added a rule not to answer status questions from them. Re-running the failing cases, the model still answered from the snippet (4 of 8 routed correctly vs 7 of 8 for the original).

**Run 3: removing the cause did.** The result snippet is now used only for ranking and never shown to the model. On the three cases that kept failing (d1, d2, f1), the shortlist went from 0 of 3 to **3 of 3** correct.

| run | variant | correct | notes |
|---|---|---:|---|
| 1 | original | 10 / 12 | misses: n1, f2 |
| 1 | shortlist (snippets shown) | 6 / 10 | 4 answered from snippet |
| 2 | shortlist (snippets labelled stale) | 4 / 8 | still answered from snippet |
| 3 | shortlist (snippets hidden) | **3 / 3** | the 3 cases that kept failing |
| 4 | original (second model) | 12 / 12 | |
| 4 | shortlist, final code (second model) | **12 / 12** | x2 retried once, see below |

**Run 4: full rerun on the final code, with a second model.** Runs 1 to 3 used an earlier version of the code, and run 3 only covered three cases. The first model stopped working on OpenRouter, so I reran all 12 cases for both variants on `nvidia/nemotron-3-ultra-550b-a55b:free`. This covers archiving and the `search_agents` fix (fix 5 is inside execution agents, which this eval doesn't exercise). Both variants got **12 of 12**, with no duplicates or wrong agents, and d1, d2 and f1 all passed. The shortlist used about **35% fewer prompt tokens** (~4.5k vs ~7.0k per call) and was slightly faster. One shortlist case (x2) first came back as `no_route` with 0 prompt tokens in 0.5 seconds, meaning the provider returned an empty response before the model read anything; rerunning it gave the correct agent. Results are in `evals/results/llm_nvidia-nemotron-3-ultra-550b-a55b-free_300_final.json` and `..._x2retry.json`.

What this says honestly: at 300 agents, this stronger model routes fine even with the full roster, so here the shortlist's win is cost and latency rather than accuracy. The original's cost keeps growing with the roster (the agent list alone is ~13k tokens at 1,000 agents), while the shortlist stays flat. Token counts differ between the two models because they count tokens differently, so compare within a run, not across runs.

Prompt size per call at 300 agents (run 1 model): ~15.8k tokens for the original vs ~7.8k for the shortlist. The saving is smaller than the agent list alone suggests, because the system prompt and tool definitions are a fixed cost.

Other things these runs showed:
- **No duplicates and no wrong agents in either variant.** The model was good at not reusing the wrong agent; its failure mode was not routing at all.
- **n1 ("Email Carla Mendes asking for the hackathon slides") fails for both variants.** The model answers instead of creating an agent, probably asking for Carla's email address. That is arguably reasonable, so this case may be mislabelled rather than a routing failure.
- **Results are noisy between runs** (p2 failed in run 1 and passed in run 2 with the same code). A real evaluation needs more cases and several runs per case.

**The lesson I'd take from this:** a cheap offline metric (is the right agent shown?) is necessary but not sufficient. It said everything was fine, and only the end-to-end eval caught that a change meant to *help* routing made the model skip routing. And when a prompt instruction fights against what's in the context, changing the context is more reliable than adding instructions.

## Limitations and what I'd do next

- **Small LLM eval.** 12 cases, one run each, on two free models, limited by the free tier. Needs all 40 cases and several runs per case before trusting the percentages, and a run at 1,000 agents, where the original should start to struggle.
- **My test cases were written by the same person who wrote the selector.** That's a real bias. Next step: sample real (anonymized) user messages and label them, or have a second person write cases blind.
- **Keyword matching can't do pure paraphrase.** Embeddings help, but the real backstop is `search_agents`, which depends on the model deciding to call it.
- **Fix 5 (execution-agent history) has no end-to-end test.** The LLM eval only exercises the interaction agent's routing, so the history change is covered by unit tests only. A next step would be an eval where an execution agent has to recall something from an old, condensed request.
- **Condensed history can lose detail.** A one-line digest drops old tool output, so a request like "forward the attachment from that email in March" might need the agent to search again. That's the trade-off for a bounded prompt; an LLM-written summary would keep more but costs a call.
- **The roster file still grows.** Archiving bounds what the model sees, not what's on disk. At real-world sizes that's a few hundred KB, so I left it; past that, archived records could move to a separate file.
- **Agent descriptions are raw instruction snippets.** A one-line LLM-written summary per agent (on a cheap model, only when the agent changes) would rank better than the first 200 characters of its instructions.

## Running it

```bash
pip install -r server/requirements.txt -r requirements-dev.txt
pip install fastembed               # optional: semantic matching
python -m pytest -q                 # unit tests
python -m evals.retrieval_eval      # retrieval eval, no API key needed
python -m evals.llm_eval            # LLM eval, needs OPENROUTER_API_KEY in .env
```

Tuning knobs (env vars): `OPENPOKE_ROUTER_TOP_K` (default 8), `OPENPOKE_ROUTER_RECENT` (default 3), `OPENPOKE_SEMANTIC_ROUTING=0` to turn off embeddings, `OPENPOKE_ROUTER_ARCHIVE_DAYS` (default 60, `0` = off), `OPENPOKE_EXEC_HISTORY_TURNS` (default 6, `0` = full history).
