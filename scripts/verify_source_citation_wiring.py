"""End-to-end (offline) check of the BUG-028 answer wiring on a REAL transcript.

Proves the grounded non-stream answer path turns a model's fragment citation into
a source deep-link at the REAL spoken moment — not the chunk head — using:
  * a REAL transcript fixture (codeaesthetic 'Dependency Injection', 349 segments),
  * a REAL chunk from it, split into REAL fragments (PR #27 helper),
  * FakeLLM.complete_json, which cites the deepest [source.fragment] tag it sees.

Also checks the safe fallbacks (no provider / no fragments → None → caller uses the
plain path) and the citation-application guardrails (out-of-range ids ignored).

Run: .venv/Scripts/python.exe scripts/verify_source_citation_wiring.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

_tmp = tempfile.mkdtemp(prefix="ytrag_cite_")
os.environ["YTRAG_TEST_MODE"] = "1"
os.environ["YTRAG_DEV_LOGIN"] = "1"
os.environ.pop("YTRAG_ENV", None)
os.environ["YTRAG_DATA_DIR"] = str(Path(_tmp) / "data")
os.environ["YTRAG_CHROMA_DIR"] = str(Path(_tmp) / "chroma")

from ytrag.chunk import chunk_transcript, fragments_for_chunk  # noqa: E402
from ytrag.config import load_config  # noqa: E402
from ytrag.llm import JsonCompletion  # noqa: E402
from ytrag.rag import (  # noqa: E402
    Assistant,
    _apply_fragment_citations,
    _format_context_tagged,
)
from ytrag.usage import Usage  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    RESULTS.append((name, passed, "" if passed else str(detail)))


FIXTURE = (Path(__file__).resolve().parents[1]
           / "e2e" / "fixtures" / "transcripts" / "J1f5b4vcxCQ.json")
doc = json.loads(FIXTURE.read_text(encoding="utf-8"))
chunk0 = chunk_transcript(doc)[0]
frags = fragments_for_chunk(doc, chunk0)  # 57 real fragments, starts 1s..~118s

# A synthetic retrieval hit for that chunk (chunk-head start, as stored today).
def _hit():
    return {
        "text": chunk0.text,
        "meta": {"video_id": doc["id"], "title": doc["title"], "url": doc["url"],
                 "index": chunk0.index, "start": chunk0.start, "author": "codeaesthetic"},
        "distance": 0.1,
    }


# --- _format_context_tagged emits [i.j] fragment tags -----------------------
ctx = _format_context_tagged([_hit()], {1: frags})
check("tagged context contains [1.0] and a deep [1.56] fragment marker",
      "[1.0]" in ctx and f"[1.{len(frags) - 1}]" in ctx, ctx[:120])

# --- end-to-end: Assistant._answer_with_citations on the real chunk ---------
cfg = load_config()
asst = Assistant(cfg, persona="test", collection="cite_test",
                 fragment_provider=lambda h: frags)
u = SimpleNamespace(standalone_query="what is dependency injection", usage=Usage())
hits = [_hit()]
res = asst._answer_with_citations(u, hits, None)

deepest = frags[-1]  # FakeLLM cites the deepest fragment tag => the last one
check("citation path returned a grounded result (not fallback)",
      res is not None and res.get("grounded") is True, res)
check("cited source's start moved to the REAL fragment timecode (not chunk head)",
      res is not None
      and res["sources"][0]["meta"]["start"] == deepest["start"]
      and deepest["start"] > chunk0.start + 30,
      f"start={res and res['sources'][0]['meta']['start']} head={chunk0.start} "
      f"deep={deepest['start']}")
check("cited source snippet is now the cited fragment text (not the chunk head)",
      res is not None and res["sources"][0]["text"] == deepest["text"],
      res and res["sources"][0]["text"][:60])
check("answer text carried through from the structured reply",
      res is not None and isinstance(res.get("answer"), str) and res["answer"].strip(),
      res and res.get("answer"))

# --- fallback: no provider -> None (caller uses the plain path) -------------
asst_np = Assistant(cfg, persona="test", collection="cite_test2",
                    fragment_provider=None)
check("no fragment_provider -> None (safe fallback to plain answer)",
      asst_np._answer_with_citations(u, [_hit()], None) is None)

# --- fallback: provider yields no fragments -> None -------------------------
asst_empty = Assistant(cfg, persona="test", collection="cite_test3",
                       fragment_provider=lambda h: [])
check("provider returns [] -> None (safe fallback)",
      asst_empty._answer_with_citations(u, [_hit()], None) is None)

# --- guardrail: bad/out-of-range citations never corrupt a source -----------
h = _hit()
applied_bad = _apply_fragment_citations([h], {1: frags},
                                        [{"source": 1, "fragment": 9999},   # bad frag
                                         {"source": 5, "fragment": 0},       # no source
                                         {"source": "x", "fragment": "y"}])  # non-int
check("out-of-range / bogus citations apply nothing (returns [])",
      applied_bad == [] and h["meta"]["start"] == chunk0.start and h["text"] == chunk0.text,
      f"applied={applied_bad} start={h['meta']['start']}")

# valid citation applies + is returned; a second for the same source is ignored
h2 = _hit()
applied_ok = _apply_fragment_citations([h2], {1: frags},
                                       [{"source": 1, "fragment": 10},
                                        {"source": 1, "fragment": 20}])
check("first valid citation wins; returns [1]; same-source duplicate ignored",
      applied_ok == [1] and h2["meta"]["start"] == frags[10]["start"], applied_ok)

# order + dedup across sources
order = _apply_fragment_citations(
    [_hit(), _hit()], {1: frags, 2: frags},
    [{"source": 2, "fragment": 0}, {"source": 2, "fragment": 1},
     {"source": 1, "fragment": 0}])
check("_apply returns applied sources in citation order, deduped", order == [2, 1], order)

# --- bad-fragment citation: keep the answer, list the source at chunk-head (F2) ---
class _StubLLM:
    def __init__(self, citations):
        self._c = citations

    def complete_json(self, system, messages, schema, **kw):
        return JsonCompletion(data={"answer": "hi", "citations": self._c},
                              usage=Usage(prompt_tokens=10, completion_tokens=5), ok=True)


asst_bad = Assistant(cfg, persona="test", collection="cite_bad",
                     fragment_provider=lambda h: frags)
asst_bad._llm = _StubLLM([{"source": 1, "fragment": 99999}])  # source ok, fragment bad
hb = _hit()
rbad = asst_bad._answer_with_citations(
    SimpleNamespace(standalone_query="q", usage=Usage()), [hb], None)
check("bad fragment: NO second full call — structured answer reused (F2)",
      rbad is not None and rbad.get("answer") == "hi", rbad)
check("bad fragment: source listed at chunk-head (not moved, not dropped)",
      rbad is not None and len(rbad["sources"]) == 1
      and rbad["sources"][0]["meta"]["start"] == chunk0.start, rbad)

# --- F1: a cited source WITHOUT fragments (e.g. a document) is not dropped ------
def _doc_hit():
    return {"text": "doc body", "meta": {"video_id": "", "title": "Notes",
            "index": 0, "start": 0.0, "author": "me"}, "distance": 0.2}


asst_mix = Assistant(cfg, persona="test", collection="cite_mix",
                     fragment_provider=lambda h: (frags if h["meta"].get("url") else []))
asst_mix._llm = _StubLLM([{"source": 1, "fragment": 4}, {"source": 2, "fragment": 0}])
mix_hits = [_hit(), _doc_hit()]  # 1 = video w/ frags, 2 = document (no frags)
rmix = asst_mix._answer_with_citations(
    SimpleNamespace(standalone_query="q", usage=Usage()), mix_hits, None)
check("F1: fragmented source moved to precise timecode AND doc source kept at head",
      rmix is not None and len(rmix["sources"]) == 2
      and rmix["sources"][0]["meta"]["start"] == frags[4]["start"]
      and rmix["sources"][1]["meta"]["start"] == 0.0,
      f"sources={[s['meta']['start'] for s in (rmix or {}).get('sources', [])]}")


# --- Sources narrowed to ONLY what the model cited ---------------------------
chunks = chunk_transcript(doc)
frags0 = fragments_for_chunk(doc, chunks[0])
frags1 = fragments_for_chunk(doc, chunks[1])


def _hit_idx(i):
    ch = chunks[i]
    return {"text": ch.text,
            "meta": {"video_id": doc["id"], "title": doc["title"], "url": doc["url"],
                     "index": ch.index, "start": ch.start, "author": "codeaesthetic"},
            "distance": 0.1 + i * 0.01}


asst2 = Assistant(cfg, persona="test", collection="cite_test_multi",
                  fragment_provider=lambda h: (frags0 if h["meta"]["index"] == 0 else frags1))
two_hits = [_hit_idx(0), _hit_idx(1)]
# The end-to-end assertion below relies on FakeLLM citing the DEEPEST [i.j] tag,
# which resolves to source 1 only while chunk0 has strictly more fragments than
# chunk1. Assert that fixture invariant so a fixture/chunker drift fails HERE with a
# clear message rather than looking like a _cited_sources regression.
check("fixture invariant: chunk0 has more fragments than chunk1",
      len(frags0) > len(frags1), f"{len(frags0)} vs {len(frags1)}")
res2 = asst2._answer_with_citations(u, two_hits, None)
# FakeLLM cites the deepest [i.j] tag → source 1 (chunk0 has the most fragments),
# so the OTHER retrieved hit must be dropped from the shown Sources.
check("Sources narrowed to only the cited source (uncited hit dropped)",
      res2 is not None and len(res2["sources"]) == 1
      and res2["sources"][0]["meta"]["index"] == 0,
      f"n={res2 and len(res2['sources'])}")



# --- REAL provider: on-disk transcript -> fragments (the #1 integration seam) --
from types import SimpleNamespace as _NS  # noqa: E402

from ytrag.bot_service import _fragment_provider_for  # noqa: E402

_corpus_id = "cid_test"
_src_id = "sid_test"
_tdir = Path(cfg.data_dir) / "corpora" / _corpus_id / "sources" / _src_id / "transcripts"
_tdir.mkdir(parents=True, exist_ok=True)
(_tdir / f"{doc['id']}.json").write_text(json.dumps(doc, ensure_ascii=False),
                                         encoding="utf-8")
provider = _fragment_provider_for(cfg, _NS(corpus_id=_corpus_id))
prov_frags = provider(_hit())
check("real provider loads on-disk transcript -> fragments matching the chunk",
      len(prov_frags) == len(frags)
      and prov_frags[0]["start"] == frags[0]["start"]
      and prov_frags[-1]["start"] == frags[-1]["start"],
      f"n={len(prov_frags)} vs {len(frags)}")
check("real provider returns [] for a document hit (no url)",
      provider({"meta": {"video_id": "x", "index": 0}}) == [])
check("real provider returns [] for an unknown video",
      provider({"meta": {"video_id": "nope", "url": "u", "index": 0}}) == [])
# alignment guard: a hit whose stored start/end don't match the reconstructed chunk
# (wrong-source same video / transcript changed since indexing) → no fragments
mismatch = _hit()
mismatch["meta"]["start"] = chunk0.start + 999.0
check("real provider returns [] on start/end mismatch (alignment guard)",
      provider(mismatch) == [], "guard did not fire")


# --- report -----------------------------------------------------------------
print()
passed = sum(1 for _, ok, _ in RESULTS if ok)
for name, ok, detail in RESULTS:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -> {detail}" if not ok else ""))
print()
print(f"real chunk head={chunk0.start}s -> cited fragment={frags[-1]['start']}s "
      f"({len(frags)} fragments)")
print(f"{passed}/{len(RESULTS)} checks passed")
print("ALL PASS" if passed == len(RESULTS) else "SOME FAILED")
sys.exit(0 if passed == len(RESULTS) else 1)
