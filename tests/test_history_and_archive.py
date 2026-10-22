"""Tests for the two extra gaps: execution-agent history growth and stale agents."""

from datetime import timedelta

from evals.synthetic_roster import NOW, build_roster
from server.agents.execution_agent.agent import render_history, split_into_turns
from server.agents.interaction_agent.agent import render_agent_list
from server.services.execution.agent_selector import is_archived, select_agents
from server.services.execution.roster import AgentRecord

T = NOW.timestamp()


def _log(n_turns, tool_output="x" * 2000):
    """A fake execution-agent log with n turns, each with a big tool response."""
    entries = []
    for i in range(n_turns):
        ts = f"2026-10-{1 + i % 28:02d}T09:00:00"
        entries += [
            ("agent_request", ts, f"Request number {i}: summarize today's inbox"),
            ("agent_action", ts, "Calling gmail_search"),
            ("tool_response", ts, f"gmail_search: {tool_output}"),
            ("agent_response", ts, f"Summary {i} sent"),
        ]
    return entries


# ---------------------------------------------------------------------- history
def test_short_history_is_unchanged():
    out = render_history(_log(3), full_turns=6)
    assert "Earlier requests" not in out
    assert out.count("<agent_request") == 3


def test_old_turns_are_condensed_and_recent_kept_in_full():
    out = render_history(_log(10), full_turns=6)
    assert out.count("<agent_request") == 6            # last 6 in full
    assert "Earlier requests (condensed, 4 total)" in out
    assert "Request number 0" in out and "Summary 0 sent" in out  # old turn still visible in one line
    assert "Request number 9" in out


def test_history_size_stays_bounded_as_agent_ages():
    sizes = [len(render_history(_log(n), full_turns=6)) for n in (10, 100, 1000)]
    assert sizes[2] < sizes[0] * 2                     # 1000 turns is not 100x bigger than 10
    assert "oldest 974 not shown" in render_history(_log(1000), full_turns=6)
    original = len(render_history(_log(1000), full_turns=0))   # 0 = original behaviour
    assert sizes[2] * 50 < original


def test_condensed_lines_drop_tool_output():
    out = render_history(_log(10, tool_output="SECRET_BULK"), full_turns=2)
    digest = out.split("## Recent requests")[0]
    assert "SECRET_BULK" not in digest


def test_turn_split():
    assert len(split_into_turns(_log(5))) == 5
    assert split_into_turns([]) == []


# ---------------------------------------------------------------------- archive
def _rec(name, task, days):
    when = (NOW - timedelta(days=days)).isoformat(timespec="seconds")
    return AgentRecord(name=name, first_task=task, last_task=task, last_used_at=when)


def test_archive_is_derived_from_last_use():
    assert is_archived(_rec("a", "x", 90), T)
    assert not is_archived(_rec("a", "x", 10), T)
    assert not is_archived(_rec("a", "x", 90), T, after_days=0)    # 0 turns it off
    assert not is_archived(AgentRecord(name="legacy"), T)          # no timestamp: never archived


def test_old_agent_still_shown_on_strong_match():
    roster = build_roster(1000) + [_rec("Passport renewal", "Renew my passport at Service Canada", 200)]
    sel = select_agents("did my passport renewal go through?", roster, now=T)
    assert sel.agents[0].name == "Passport renewal"


def test_old_weak_matches_stop_taking_slots():
    fresh = _rec("Email to Dana re soccer schedule", "Email Dana about the soccer schedule", 3)
    # old agents that only share one word ("Dana") with the message: weak matches
    old = [_rec(f"Dana note {i}", f"Note about Dana's plant {i}", 120) for i in range(10)]
    roster = [_rec(f"Filler {i}", f"Unrelated errand {i}", 5) for i in range(100)] + [fresh] + old
    on = select_agents("soccer schedule with Dana", roster, now=T, recent=0)
    off = select_agents("soccer schedule with Dana", roster, now=T, recent=0, archive_after_days=0)
    assert on.agents[0].name == fresh.name
    assert len(on.agents) < len(off.agents)            # weak old matches filtered out
    assert on.archived == 10


def test_rendered_list_mentions_archived_count():
    roster = build_roster(300)
    text = render_agent_list(select_agents("Tokyo flights", roster, now=T))
    assert "archived" in text


def test_search_agents_never_returns_old_results(tmp_path, monkeypatch):
    """Same stale-answer bug as the shortlist: the search tool must not show last_result."""
    from server.agents.interaction_agent import tools
    from server.services.execution.roster import AgentRoster

    roster = AgentRoster(tmp_path / "roster.json")
    rec = AgentRecord(name="Lunch thread", first_task="Ask Alice about lunch",
                      last_result="Alice replied Thursday works", last_used_at=NOW.isoformat())
    roster._agents[rec.name] = rec
    monkeypatch.setattr(tools, "get_agent_roster", lambda: roster)
    monkeypatch.setattr(roster, "load", lambda: None)

    result = tools.search_agents("Alice lunch")
    assert result.payload["matches"][0]["agent_name"] == "Lunch thread"
    assert "Thursday" not in str(result.payload)
