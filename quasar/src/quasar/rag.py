"""Retrieval over the SOP corpus, and the grounding check that makes it load-bearing.

Retrieval is Okapi BM25, implemented here rather than pulled in, for two reasons.
It is exactly reproducible -- the same query returns the same sections in the
same order on match day as it did in the tabletop exercise, which an approximate
nearest-neighbour index over a floating-point embedding space does not guarantee
-- and it is inspectable: an operator can be shown *why* a section was retrieved.
A dense retriever can be layered in front of it (``Retriever`` takes an optional
reranker), but the lexical index remains the floor, so a network partition
degrades retrieval quality rather than removing it.

The second half of this module is the part that actually matters. Retrieval
grounds a model; it does not *make* the model grounded. :func:`check_grounding`
verifies, after generation, that every SOP reference the model emitted names a
real section that was actually in the retrieved context. A brief that cites
SOP-EVAC-01#9 -- which does not exist -- or cites a section the model was never
shown is rejected by the governance layer and replaced by the deterministic
fallback. This is the difference between a system that cites and a system that
can be trusted to have read what it cites.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from quasar.sops import SOP_BY_CATEGORY, SOP_CORPUS, SOP_INDEX, SopSection, sections_of

_TOKEN = re.compile(r"[a-z0-9]+")

# Words that carry no discriminative weight in an operational corpus where every
# document is about the venue.
_STOPWORDS = frozenset(
    """a an the is are was were be been being of to in on at by for with from and or not no
    any all this that these those it its as if then than must may shall should will would can
    each per into out up down over under""".split()
)

BM25_K1 = 1.5
BM25_B = 0.75


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


@dataclass(frozen=True, slots=True)
class Retrieved:
    section: SopSection
    score: float

    @property
    def ref(self) -> str:
        return self.section.ref


class Retriever:
    """BM25 over SOP sections, with an optional reranker in front."""

    def __init__(
        self,
        corpus: Sequence[SopSection] = SOP_CORPUS,
        *,
        reranker: Callable[[str, Sequence[Retrieved]], Sequence[Retrieved]] | None = None,
    ) -> None:
        if not corpus:
            raise ValueError("corpus is empty")
        self._corpus = tuple(corpus)
        self._reranker = reranker

        self._docs: list[Counter[str]] = []
        self._lengths: list[int] = []
        df: Counter[str] = Counter()
        for section in self._corpus:
            terms = Counter(tokenize(f"{section.title} {section.text}"))
            self._docs.append(terms)
            self._lengths.append(sum(terms.values()))
            df.update(terms.keys())

        n = len(self._corpus)
        self._avg_len = sum(self._lengths) / n
        # Robertson/Sparck-Jones IDF with the +0.5 smoothing, floored at a small
        # positive value so a term appearing in every section cannot drive a
        # score negative and invert the ranking.
        self._idf: Mapping[str, float] = {
            term: max(math.log((n - freq + 0.5) / (freq + 0.5) + 1.0), 1e-6)
            for term, freq in df.items()
        }

    def search(
        self, query: str, *, k: int = 4, pinned_docs: Sequence[str] = ()
    ) -> tuple[Retrieved, ...]:
        """BM25 over the corpus, with the governing documents always included.

        ``pinned_docs`` are placed in context unconditionally. Grounding is strict
        -- an agent may only cite what it was shown -- so a retriever that misses
        the clause the model correctly applied causes a *correct* brief to be
        rejected. Pinning the document that procedure says governs this incident
        category takes retrieval recall off the safety path; BM25 is then free to
        do the thing it is actually good at, which is surfacing the additional,
        non-obvious sections that also bear on the situation.
        """
        if k <= 0:
            raise ValueError("k must be positive")

        pinned: list[Retrieved] = []
        pinned_refs: set[str] = set()
        for doc_id in pinned_docs:
            for section in sections_of(doc_id):
                pinned.append(Retrieved(section=section, score=math.inf))
                pinned_refs.add(section.ref)
        if not pinned and not tokenize(query):
            return ()

        q_terms = tokenize(query)
        scored: list[Retrieved] = []
        for i, section in enumerate(self._corpus):
            if section.ref in pinned_refs:
                continue
            terms, length = self._docs[i], self._lengths[i]
            score = 0.0
            for term in q_terms:
                tf = terms.get(term, 0)
                if tf == 0:
                    continue
                norm = 1.0 - BM25_B + BM25_B * (length / self._avg_len)
                score += self._idf[term] * (tf * (BM25_K1 + 1.0)) / (tf + BM25_K1 * norm)
            if score > 0.0:
                scored.append(Retrieved(section=section, score=score))

        # Ties break on ref so the retrieved context is byte-stable, which in
        # turn keeps the prompt cache warm across the match.
        scored.sort(key=lambda r: (-r.score, r.ref))

        pinned.sort(key=lambda r: r.ref)
        remaining = max(0, k - len(pinned))
        top = tuple(pinned) + tuple(scored[:remaining])

        if self._reranker is not None and scored:
            reranked = tuple(self._reranker(query, scored))[:remaining]
            top = tuple(pinned) + reranked
        return top

    def for_incident(self, category: str, text: str, *, k: int = 8) -> tuple[Retrieved, ...]:
        """Context for an incident brief: the governing SOPs, plus what the words suggest."""
        return self.search(text, k=k, pinned_docs=SOP_BY_CATEGORY.get(category, ()))


def render_context(hits: Sequence[Retrieved]) -> str:
    """Format retrieved sections for the model prompt, with their refs attached.

    The ref is rendered *with* the text rather than in a separate list so the
    model cannot cite a section whose content it was not shown.
    """
    return "\n\n".join(
        f"[{h.ref}] {h.section.title}\n{h.section.text}" for h in hits
    )


@dataclass(frozen=True, slots=True)
class GroundingResult:
    ok: bool
    cited: tuple[str, ...]
    unknown: tuple[str, ...]  # cited refs that do not exist in the corpus
    unretrieved: tuple[str, ...]  # cited refs that exist but were not in context

    @property
    def reason(self) -> str:
        parts: list[str] = []
        if self.unknown:
            parts.append(f"cites non-existent SOP sections: {', '.join(self.unknown)}")
        if self.unretrieved:
            parts.append(
                "cites SOP sections that were not in the retrieved context: "
                + ", ".join(self.unretrieved)
            )
        if not self.cited:
            parts.append("cites nothing")
        return "; ".join(parts)


def check_grounding(
    citations: Sequence[Mapping[str, str]],
    retrieved: Sequence[Retrieved],
    *,
    require_retrieved: bool = True,
) -> GroundingResult:
    """Verify a payload's citations against the corpus and the retrieved context.

    ``require_retrieved`` is the strict mode used for incident briefs: the model
    may only cite what it was actually shown. It is relaxed for volunteer
    briefings, which may legitimately reference standing procedure the retriever
    did not surface for this particular query -- but even then the section must
    exist.
    """
    refs = tuple(f"{c['doc_id']}#{c['section']}" for c in citations)
    in_context = {r.ref for r in retrieved}

    unknown = tuple(r for r in refs if r not in SOP_INDEX)
    unretrieved = (
        tuple(r for r in refs if r in SOP_INDEX and r not in in_context)
        if require_retrieved
        else ()
    )
    ok = bool(refs) and not unknown and not unretrieved
    return GroundingResult(ok=ok, cited=refs, unknown=unknown, unretrieved=unretrieved)
