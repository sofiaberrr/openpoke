"""Unit tests for the agent-overload fix. Run with:  python -m pytest -q

No API key or network needed: the selector runs keyword-only here (semantic matching
is switched off so results are deterministic).
"""

import asyncio
import json
import os

os.environ["OPENPOKE_SEMANTIC_ROUTING"] = "0"

import pytest

from evals.synthetic_roster import NOW, build_roster
from server.agents.interaction_agent.agent import render_agent_list
from server.services.execution.agent_selector import select_agents, tokenize
from server.services.execution.roster import AgentRecord, AgentRoster, normalize_name

T = NOW.timestamp()


# ---------------------------------------------------------------------- roster
def test_old_roster_file_is_upgraded(tmp_path):
    path = tmp_path / "roster.json"
    path.write_text(json.dumps(["Email to Alice", "Tokyo trip"]))
    roster = AgentRoster(path)
    assert roster.get_agents() == ["Email to Alice", "Tokyo trip"]
    roster.record_request("Email to Alice", "Ask Alice about lunch")
    saved = json.loads(path.read_text())
    assert saved[0]["name"] == "Email to Alice"
    assert saved[0]["last_task"] == "Ask Alice about lunch"
    assert saved[0]["use_count"] == 1


def test_record_request_and_result_update_the_record(tmp_path):
    roster = AgentRoster(tmp_path / "roster.json")
    roster.add_agent("Email to Bob", "Ask Bob about soccer")
    roster.record_request("Email to Bob", "Ask Bob to bring the ball")
    roster.record_result("Email to Bob", "Bob says yes")
    rec = AgentRoster(tmp_path / "roster.json").get("Email to Bob")  # reload from disk
    assert rec.first_task == "Ask Bob about soccer"
    assert rec.last_task == "Ask Bob to bring the ball"
    assert rec.last_result == "Bob says yes"
    assert rec.use_count == 1


@pytest.mark.parametrize("new,existing,expected", [
    ("Email Alice", "Email to Alice", "Email to Alice"),           # filler word
    ("alice email", "Email to Alice", "Email to Alice"),           # word order / case
    ("Email to Alcie", "Email to Alice", "Email to Alice"),        # typo in a long word
    ("Email to Bob", "Email to Alice", None),                      # different person
    ("Email to Rob", "Email to Bob", None),                        # short names are never "typos"
    ("Email to Alice re budget", "Email to Alice", None),          # different thread, keep separate
])
def test_duplicate_guard(tmp_path, new, existing, expected):
    roster = AgentRoster(tmp_path / "roster.json")
    roster.add_agent(existing)
    assert roster.find_duplicate(new) == expected


def test_normalize_name():
    assert normalize_name("Email to Alice Chen") == normalize_name("alice chen email")


def test_tokenize_drops_filler_and_plurals():
    assert tokenize("Did the flights to Tokyo get booked?") == ["flight", "tokyo", "booked"]


# ---------------------------------------------------------------------- selector
def test_shortlist_is_small_even_for_huge_rosters():
    roster = build_roster(1000)
    sel = select_agents("Did Alice Chen reply about lunch?", roster, now=T, top_k=8, recent=3)
    assert sel.total == 1000
    assert len(sel.agents) <= 11
    assert sel.agents[0].name == "Email to Alice Chen re lunch"


def test_picks_right_alice_among_distractors():
    roster = build_roster(300)
    sel = select_agents("Reply to Alice Park about the budget", roster, now=T)
    assert sel.agents[0].name == "Email to Alice Park re Q3 budget"


def test_follow_up_uses_conversation_context():
    roster = build_roster(300)
    ctx = "<user_message>When do I fly out to Tokyo?</user_message><poke_reply>March 12.</poke_reply>"
    sel = select_agents("and the return flight?", roster, context=ctx, now=T)
    assert "Tokyo trip flights" in [a.name for a in sel.agents]


def test_recent_agents_always_included():
    records = [
        AgentRecord(name="Old unrelated", first_task="x", last_used_at="2026-01-01T00:00:00+00:00"),
        AgentRecord(name="Just used", first_task="y", last_used_at="2026-10-17T11:00:00+00:00"),
    ]
    sel = select_agents("something with no matches at all", records, now=T, top_k=5, recent=1)
    assert [a.name for a in sel.agents] == ["Just used"]
    assert sel.reasons["Just used"] == "recent"


def test_recency_breaks_ties():
    a = AgentRecord(name="Email to Sam A", first_task="email Sam about dinner", last_used_at="2026-05-01T00:00:00+00:00")
    b = AgentRecord(name="Email to Sam B", first_task="email Sam about dinner", last_used_at="2026-10-16T00:00:00+00:00")
    sel = select_agents("email Sam about dinner", [a, b], now=T, recent=0)
    assert sel.agents[0].name == "Email to Sam B"


def test_rendered_list_tells_model_about_hidden_agents():
    roster = build_roster(300)
    text = render_agent_list(select_agents("Tokyo flights", roster, now=T))
    assert 'total="300"' in text
    assert "search_agents" in text
    assert "Tokyo trip flights" in text


def test_results_are_used_for_ranking_but_never_shown():
    """Regression test for the bug the LLM eval caught (answering from stale snippets)."""
    rec = AgentRecord(name="Lunch thread", first_task="Ask about lunch",
                      last_result="Alice replied Thursday works", last_used_at="2026-10-16T00:00:00+00:00")
    sel = select_agents("did Alice reply", [rec], now=T, recent=0)
    assert sel.agents and sel.agents[0].name == "Lunch thread"  # result text helped it match
    assert "Thursday" not in render_agent_list(sel)               # but the model never sees it


# ---------------------------------------------------------------------- tools
def test_search_agents_tool_finds_hidden_agents(tmp_path, monkeypatch):
    from server.agents.interaction_agent import tools

    roster = AgentRoster(tmp_path / "roster.json")
    for rec in build_roster(200):
        roster._agents[rec.name] = rec
    monkeypatch.setattr(tools, "get_agent_roster", lambda: roster)
    monkeypatch.setattr(roster, "load", lambda: None)

    result = tools.search_agents("Kyoto ryokan")
    assert result.payload["searched"] == 200
    assert result.payload["matches"][0]["agent_name"] == "Kyoto ryokan booking"


def test_send_message_merges_near_duplicate_name(tmp_path, monkeypatch):
    from server.agents.interaction_agent import tools

    roster = AgentRoster(tmp_path / "roster.json")
    roster.add_agent("Email to Alice", "lunch")
    monkeypatch.setattr(tools, "get_agent_roster", lambda: roster)
    monkeypatch.setattr(tools.get_execution_agent_logs(), "record_request", lambda *a: None)

    async def fake_execute(*a, **k):
        return None
    monkeypatch.setattr(tools._EXECUTION_BATCH_MANAGER, "execute_agent", fake_execute)

    async def run():
        return tools.send_message_to_agent("email alice", "did she reply?")
    result = asyncio.run(run())
    assert result.payload["agent_name"] == "Email to Alice"
    assert result.payload["new_agent_created"] is False
    assert result.payload["merged_from"] == "email alice"
    assert roster.get_agents() == ["Email to Alice"]


# ---------------------------------------------------------------------- llm eval harness
def test_llm_eval_grading_with_fake_model(monkeypatch):
    """The LLM eval harness itself works (a scripted model, no network)."""
    from evals import llm_eval
    from evals.cases import CASES

    def fake_reply(name, args):
        return {"choices": [{"message": {"content": "", "tool_calls": [
            {"id": "1", "function": {"name": name, "arguments": json.dumps(args)}}]}}],
            "usage": {"prompt_tokens": 100}}

    script = iter([
        fake_reply("search_agents", {"query": "Kyoto hotel"}),
        fake_reply("send_message_to_agent", {"agent_name": "Kyoto ryokan booking", "instructions": "check"}),
    ])

    async def fake_completion(**kwargs):
        return next(script)
    monkeypatch.setattr(llm_eval, "request_chat_completion", fake_completion)
    monkeypatch.setattr(llm_eval, "PAUSE_SECONDS", 0)

    case = next(c for c in CASES if c.id == "x5")
    row = asyncio.run(llm_eval.run_case(case, "shortlist", build_roster(100), "fake", "key"))
    assert row["outcome"] == "correct"
    assert row["searched"] is True
    assert row["calls"] == 2
