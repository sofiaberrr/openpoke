"""Pick which execution agents the interaction agent gets to see on each turn.

The problem ("agent overload"): agents are never deleted, so after weeks of use the
roster has hundreds of entries. Sending all of them on every turn costs tokens and
latency, and makes the model worse at picking the right one (or noticing that one
already exists, so it spawns duplicates).

The fix: show a short, ranked list instead of the whole roster.

    shortlist = top-K agents by relevance to the current message
              + the few most recently used agents ("hot cache", for follow-ups
                like "did she reply?" that name nothing)

Relevance is a mix of:
  * keyword match (BM25) against the agent's name, first task, last task and last
    result. Cheap, no model, deterministic, and great at names ("Alice", "Tokyo").
  * optional semantic match (local embedding model via `fastembed`) for paraphrases
    ("my landlord" -> "Email to Mark about the lease"). Off if not installed.
  * a recency boost, so a fresh agent beats a stale one when both match.

Anything not in the shortlist is still reachable through the `search_agents` tool,
so the cap never makes an agent permanently invisible.
"""

from __future__ import annotations

import math
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .roster import AgentRecord
from ...logging_config import logger

DEFAULT_TOP_K = int(os.getenv("OPENPOKE_ROUTER_TOP_K", "8"))
DEFAULT_RECENT = int(os.getenv("OPENPOKE_ROUTER_RECENT", "3"))
RECENCY_HALF_LIFE_DAYS = 14.0
# Agents unused for this long are "archived": they only make the shortlist on a strong
# match, so weak matches from months ago stop taking slots. 0 turns archiving off.
ARCHIVE_AFTER_DAYS = float(os.getenv("OPENPOKE_ROUTER_ARCHIVE_DAYS", "60"))
ARCHIVED_MIN_SCORE = 0.5

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "can", "could", "did", "do", "does", "for",
    "from", "get", "go", "had", "has", "have", "he", "her", "him", "his", "how", "i", "if",
    "in", "is", "it", "its", "just", "let", "me", "my", "of", "on", "or", "our", "please",
    "she", "so", "that", "the", "their", "them", "then", "there", "they", "this", "to", "up",
    "us", "was", "we", "what", "when", "where", "which", "who", "will", "with", "would",
    "you", "your", "yet", "any", "about", "again", "also", "back", "know", "new", "now",
    "still", "tell", "thanks", "hey", "hi", "ok", "okay", "user", "message", "agent",
    "new_user_message", "new_agent_message", "agent_message", "user_message", "assistant",
}


def tokenize(text: str) -> List[str]:
    """Lowercase words, drop filler words, strip a trailing plural 's'."""
    tokens = []
    for raw in re.findall(r"[a-z0-9]+", (text or "").lower()):
        if raw in _STOPWORDS or len(raw) < 2:
            continue
        if len(raw) > 3 and raw.endswith("s") and not raw.endswith("ss"):
            raw = raw[:-1]
        tokens.append(raw)
    return tokens


class _BM25:
    """Tiny BM25 implementation (the standard search-engine keyword score)."""

    def __init__(self, docs: Sequence[List[str]], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self.doc_freqs = [Counter(d) for d in docs]
        self.avg_len = (sum(len(d) for d in docs) / len(docs)) if docs else 0.0
        df: Counter = Counter()
        for d in docs:
            df.update(set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: List[str]) -> List[float]:
        out = []
        for freqs, doc in zip(self.doc_freqs, self.docs):
            s = 0.0
            norm = self.k1 * (1 - self.b + self.b * len(doc) / (self.avg_len or 1))
            for t in set(query):
                f = freqs.get(t)
                if f:
                    s += self.idf.get(t, 0.0) * f * (self.k1 + 1) / (f + norm)
            out.append(s)
        return out


class _SemanticScorer:
    """Optional embedding similarity. Silently disabled if unavailable."""

    _instance: Optional["_SemanticScorer"] = None

    def __init__(self) -> None:
        self.model = None
        self._cache: Dict[str, List[float]] = {}
        if os.getenv("OPENPOKE_SEMANTIC_ROUTING", "1") == "0":
            return
        try:
            from fastembed import TextEmbedding  # type: ignore

            self.model = TextEmbedding("BAAI/bge-small-en-v1.5")
            logger.info("Semantic agent routing enabled (bge-small-en-v1.5)")
        except Exception as exc:  # not installed, or model download failed
            logger.info(f"Semantic agent routing disabled: {exc}")
            self.model = None

    @classmethod
    def get(cls) -> "_SemanticScorer":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def enabled(self) -> bool:
        return self.model is not None

    def _embed(self, texts: List[str]) -> List[List[float]]:
        missing = [t for t in texts if t not in self._cache]
        if missing:
            for text, vec in zip(missing, self.model.embed(missing)):
                self._cache[text] = [float(x) for x in vec]
        return [self._cache[t] for t in texts]

    def scores(self, query: str, docs: List[str]) -> List[float]:
        q, *d = self._embed([query] + docs)
        return [max(0.0, sum(a * b for a, b in zip(q, v))) for v in d]  # vectors are normalized


@dataclass
class Selection:
    """Result of one routing decision, kept for evals and debugging."""

    agents: List[AgentRecord]          # what the interaction agent will see, best first
    total: int                         # size of the full roster
    reasons: Dict[str, str]            # agent name -> "relevant" / "recent"
    scores: Dict[str, float]
    archived: int = 0                  # how many agents in the roster are archived


def is_archived(record: AgentRecord, now: Optional[float] = None, after_days: float = ARCHIVE_AFTER_DAYS) -> bool:
    """Archived = not used for `after_days`. Derived from last_used_at, so using an
    agent again un-archives it automatically and nothing extra is stored."""
    if after_days <= 0 or not record.last_used_ts():
        return False
    now = time.time() if now is None else now
    return (now - record.last_used_ts()) / 86400.0 >= after_days


def _recency(record: AgentRecord, now: float) -> float:
    age_days = max(0.0, (now - record.last_used_ts()) / 86400.0)
    return 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS)


def rank_agents(
    query: str,
    records: Sequence[AgentRecord],
    *,
    context: str = "",
    now: Optional[float] = None,
    use_semantic: Optional[bool] = None,
) -> List[tuple]:
    """Score every agent against the message. Returns [(record, score)] best first.

    Only agents with some real match (keyword or semantic) get a nonzero score;
    recency only breaks ties between matching agents, it never makes an unrelated
    agent look relevant.
    """
    if not records:
        return []
    now = time.time() if now is None else now
    docs = [tokenize(r.search_text()) + tokenize(r.name) for r in records]  # name counts double
    bm25 = _BM25(docs)

    lexical = bm25.scores(tokenize(query))
    if context:
        ctx = bm25.scores(tokenize(context))
        lexical = [a + 0.3 * b for a, b in zip(lexical, ctx)]  # recent conversation counts a bit
    top = max(lexical) or 1.0
    lexical = [s / top for s in lexical]

    semantic = [0.0] * len(records)
    semantic_on = _SemanticScorer.get().enabled if use_semantic is None else use_semantic
    if semantic_on and _SemanticScorer.get().enabled:
        raw = _SemanticScorer.get().scores(query, [r.search_text() for r in records])
        # bge similarities sit around 0.4-0.9; rescale so only clear matches count
        semantic = [max(0.0, (s - 0.55) / 0.35) for s in raw]

    ranked = []
    for record, lex, sem in zip(records, lexical, semantic):
        match = max(lex, 0.8 * sem) + 0.2 * min(lex, sem)
        if match <= 0.05:
            continue
        ranked.append((record, match * (0.8 + 0.2 * _recency(record, now))))
    ranked.sort(key=lambda pair: pair[1], reverse=True)
    return ranked


def select_agents(
    query: str,
    records: Sequence[AgentRecord],
    *,
    context: str = "",
    top_k: int = DEFAULT_TOP_K,
    recent: int = DEFAULT_RECENT,
    now: Optional[float] = None,
    use_semantic: Optional[bool] = None,
    archive_after_days: float = ARCHIVE_AFTER_DAYS,
) -> Selection:
    """Build the shortlist the interaction agent sees this turn."""
    now = time.time() if now is None else now
    ranked = rank_agents(query, records, context=context, now=now, use_semantic=use_semantic)
    archived = {r.name for r in records if is_archived(r, now, archive_after_days)}
    # archived agents still compete, but only on a strong match
    ranked = [(r, s) for r, s in ranked if r.name not in archived or s >= ARCHIVED_MIN_SCORE]
    chosen: List[AgentRecord] = []
    reasons: Dict[str, str] = {}
    scores: Dict[str, float] = {}

    for record, score in ranked[:top_k]:
        chosen.append(record)
        reasons[record.name] = "relevant"
        scores[record.name] = round(score, 3)

    by_recency = sorted(records, key=lambda r: r.last_used_ts(), reverse=True)
    for record in by_recency[:recent]:
        if record.name not in reasons:
            chosen.append(record)
            reasons[record.name] = "recent"

    return Selection(agents=chosen, total=len(records), reasons=reasons, scores=scores,
                     archived=len(archived))


def humanize_age(record: AgentRecord, now: Optional[float] = None) -> str:
    ts = record.last_used_ts()
    if not ts:
        return "unknown"
    days = ((time.time() if now is None else now) - ts) / 86400.0
    if days < 1 / 24:
        return "just now"
    if days < 1:
        return f"{int(days * 24)}h ago"
    if days < 60:
        return f"{int(days)}d ago"
    return f"{int(days / 30)}mo ago"
