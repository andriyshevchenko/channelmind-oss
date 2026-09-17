"""Confirms the BUG-028 approach on a REAL video transcript + a REAL chunk.

BUG-028: a source's timecode/deep-link points at the START of the whole ~350-word
chunk, not the actually-cited passage, so on short videos it's a near-useless
"0:01"/"0:05". The fix: split each retrieved chunk into id-tagged caption
fragments (``fragments_for_chunk``) whose timecodes are the transcript's OWN
segment starts; the LLM cites a fragment id and we map it back to that real
moment (``timecode_for_fragment``). This validates the mechanism deterministically
— no LLM, no network — on a real transcript fetched from prod.

Fixture: e2e/fixtures/transcripts/J1f5b4vcxCQ.json — codeaesthetic's
"Dependency Injection, The Best Pattern" (349 real caption segments, ~13 min),
imported through the real ingest pipeline on the VPS.

Run: .venv/Scripts/python.exe scripts/verify_source_timecode.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ytrag.chunk import (  # noqa: E402
    chunk_transcript,
    fragments_for_chunk,
    timecode_for_fragment,
)

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    RESULTS.append((name, passed, "" if passed else str(detail)))


FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "e2e" / "fixtures" / "transcripts" / "J1f5b4vcxCQ.json"
)
doc = json.loads(FIXTURE.read_text(encoding="utf-8"))

# --- real-data anchor -------------------------------------------------------
check("fixture is the real video (id/title/segments present)",
      doc.get("id") == "J1f5b4vcxCQ"
      and "Dependency Injection" in (doc.get("title") or "")
      and len(doc.get("segments") or []) > 300,
      f"id={doc.get('id')} segs={len(doc.get('segments') or [])}")

# --- build a REAL chunk from the REAL transcript ----------------------------
chunks = chunk_transcript(doc)  # default 350 words / 60 overlap — the prod config
check("chunk_transcript produced multiple real chunks", len(chunks) >= 2,
      f"n={len(chunks)}")

chunk0 = chunks[0]
frags = fragments_for_chunk(doc, chunk0)

check("chunk splits into many timed fragments (not one)", len(frags) > 5,
      f"n={len(frags)}")
check("every fragment timecode lies within the chunk's [start,end] window",
      all(chunk0.start <= f["start"] <= chunk0.end for f in frags),
      f"start={chunk0.start} end={chunk0.end}")
check("fragment timecodes are non-decreasing (chronological)",
      all(frags[i]["start"] <= frags[i + 1]["start"] for i in range(len(frags) - 1)),
      "out of order")

# --- THE core proof: a cited fragment resolves to its OWN moment, not the head
head_ts = chunk0.start
mid = frags[len(frags) // 2]
resolved = timecode_for_fragment(frags, mid["id"])

check("chunk HEAD is near-zero (this is the misleading '0:0X' today)",
      head_ts < 10.0, f"head={head_ts}")
check("a mid-chunk cited fragment resolves to a MUCH later real moment",
      resolved is not None and resolved >= head_ts + 30.0,
      f"head={head_ts} resolved={resolved}")
check("id -> timecode round-trips to the fragment's own start",
      resolved == mid["start"], f"resolved={resolved} frag={mid['start']}")

# --- the resolved timecode is the transcript's OWN value, not computed -------
seg_match = next(
    (s for s in doc["segments"]
     if float(s.get("start", -1)) == mid["start"]
     and str(s.get("text", "")).strip() == mid["text"]),
    None,
)
check("resolved timecode is a genuine caption segment start (not computed)",
      seg_match is not None,
      f"no segment at start={mid['start']} with matching text")

# --- validation / safe fallback (garbled or hallucinated ids) ---------------
check("out-of-range fragment id -> None (caller falls back to chunk head)",
      timecode_for_fragment(frags, 10_000) is None)
check("non-integer fragment id -> None (safe)",
      timecode_for_fragment(frags, "not-an-int") is None
      and timecode_for_fragment(frags, None) is None)

# --- a later chunk also refines (not just chunk 0) --------------------------
c1 = chunks[1]
f1 = fragments_for_chunk(doc, c1)
check("a later chunk (non-zero head) also yields refined fragments past its head",
      len(f1) > 5 and f1[-1]["start"] > c1.start,
      f"c1.start={c1.start} last_frag={f1[-1]['start'] if f1 else None}")


# --- report -----------------------------------------------------------------
print()
passed = sum(1 for _, ok, _ in RESULTS if ok)
for name, ok, detail in RESULTS:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -> {detail}" if not ok else ""))
print()
print(f"real chunk[0]: head={chunk0.start}s, {len(frags)} fragments, "
      f"cited-fragment moment={mid['start']}s (id {mid['id']})")
print(f"{passed}/{len(RESULTS)} checks passed")
print("ALL PASS" if passed == len(RESULTS) else "SOME FAILED")
sys.exit(0 if passed == len(RESULTS) else 1)
