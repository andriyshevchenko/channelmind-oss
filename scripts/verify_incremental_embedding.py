"""Offline proof for BUG-021: the routine sync path embeds ONLY new videos and
never drops + re-embeds the whole corpus, while a genuine FULL rebuild still drops
and re-embeds everything — and both process videos NEWEST-FIRST.

Drives the REAL ``bot_service.rebuild_bot`` pipeline (chunk -> embed -> Chroma) with
a recording fake embedder and a drop-tracking VectorStore, over transcripts seeded
through the real ingest seam. No network, credentials, or yt-dlp.

What is checked:
  (a) INCREMENTAL add: a re-import that gained one new video embeds ONLY that
      video's chunks, never calls ``store.drop()``, and preserves every existing
      chunk (count grows by exactly the new chunks).
  (b) FULL rebuild: drops the collection and re-embeds EVERY video (the semantically
      required reconcile path — source removal / explicit Rebuild).
  (c) ORDER: both paths process videos newest-first (by ``upload_date``), not
      alphabetically by video id.
  (d) A cancelled INCREMENTAL rebuild does NOT drop the collection — existing
      vectors survive (cancelling a sync must never wipe a working bot); a cancelled
      FULL rebuild does drop (its prior vectors were already gone).

Run: .venv/Scripts/python.exe scripts/verify_incremental_embedding.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

_tmp = Path(tempfile.mkdtemp(prefix="inc_embed_"))
os.environ["YTRAG_ENV"] = "development"
os.environ["YTRAG_DEV_LOGIN"] = "1"
os.environ["YTRAG_DATA_DIR"] = str(_tmp / "data")
os.environ["YTRAG_CHROMA_DIR"] = str(_tmp / "chroma")
os.environ["YTRAG_SESSION_SECRET"] = "test-secret-inc-embed"
os.environ.pop("YTRAG_TEST_MODE", None)  # exercise the REAL chunk/embed pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import ytrag.store as store_mod  # noqa: E402
from ytrag import bot_service  # noqa: E402
from ytrag.config import load_config  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


# ---- video fixtures: ids chosen so alphabetical != chronological ----------
# id order (asc):  amiddle < mnewest < zoldest
# date order (new->old): mnewest (0601) > amiddle (0301) > zoldest (0101)
# so a newest-first-by-DATE result must be [mnewest, amiddle, zoldest] — which is
# neither the id-ascending nor id-descending order, proving the sort is by date.
V_OLD = ("zoldest", "20240101")
V_MID = ("amiddle", "20240301")
V_NEW = ("mnewest", "20240601")
V_EXTRA = ("extra4", "20240901")  # newest of all; added later to test cancel path
# A deliberately MULTI-chunk video (long transcript) for the F1 partial-heal test,
# plus a single-chunk control on the same bot.
V_MULTI = ("vmulti", "20240501")
V_SOLO = ("vsolo", "20240201")
V_SHRINK = ("vshrink", "20240701")  # transcript later shortened (present > expected)
ALL_IDS = [V_OLD[0], V_MID[0], V_NEW[0], V_EXTRA[0], V_MULTI[0], V_SOLO[0], V_SHRINK[0]]

# ---- instrumentation ------------------------------------------------------
EMBED_LOG: list[str] = []      # every text passed to the embedder, in order
DROPPED: list[str] = []        # collection names on which drop() was called
ADD_CALLS: list[set] = []      # the set of video ids in each store.add() batch
ABORT = {"arm": False, "fire": False}


class RecordingEmbedder:
    """Records the texts it embeds (order preserved) so the test can assert which
    videos were embedded and in what order. Arms the cancel probe after the first
    embed batch so scenario (d) can hit the mid-rebuild abort guard."""

    def embed_documents(self, texts, batch_size: int = 128):
        EMBED_LOG.extend(texts)
        if ABORT["arm"]:
            ABORT["fire"] = True
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text):
        return [0.1, 0.2, 0.3]


_orig_drop = store_mod.VectorStore.drop
_orig_add = store_mod.VectorStore.add


def _tracking_drop(self):
    DROPPED.append(self._name)
    return _orig_drop(self)


def _tracking_add(self, chunks, embeddings):
    ADD_CALLS.append({c.video_id for c in chunks})
    return _orig_add(self, chunks, embeddings)


def _vid_of(text: str) -> str:
    for vid in ALL_IDS:
        if vid in text:
            return vid
    return "?"


def _embed_video_order() -> list[str]:
    """The sequence of distinct video ids in embed order (consecutive dupes collapsed)."""
    order: list[str] = []
    for text in EMBED_LOG:
        vid = _vid_of(text)
        if not order or order[-1] != vid:
            order.append(vid)
    return order


def _embedded_video_set() -> set[str]:
    return {_vid_of(t) for t in EMBED_LOG}


def _seed(cfg, bot, source, vid: str, date: str, reps: int = 12) -> None:
    """Write one transcript via the REAL ingest seam (as a browser client would).

    ``reps`` scales the transcript length: the default is a single chunk; a large
    value yields a multi-chunk video (for the partial-heal test)."""
    text = f"A transcript about the {vid} topic with enough words to form a chunk. " * reps
    bot_service.ingest_text_items(
        cfg, bot, source,
        [{"id": vid, "title": f"Title {vid}", "url": f"https://youtu.be/{vid}",
          "upload_date": date, "text": text}],
    )


def _expected_chunks(cfg, bot, source, vid: str) -> int:
    """Expected chunk count for ``vid`` straight from its on-disk transcript."""
    from ytrag.chunk import chunk_transcript
    tdir = bot_service.source_transcripts_dir(cfg, bot.corpus_id, source.id)
    doc = json.loads((tdir / f"{vid}.json").read_text(encoding="utf-8"))
    return len(chunk_transcript(doc))


def _chunk_counts(cfg, bot) -> dict:
    corpus = bot_service.corpus_store(cfg).get(bot.corpus_id)
    return store_mod.VectorStore(cfg.chroma_dir, corpus.collection).video_chunk_counts()


def _collection_video_ids(cfg, bot) -> set[str]:
    return set(_chunk_counts(cfg, bot))


def _collection_count(cfg, bot) -> int:
    corpus = bot_service.corpus_store(cfg).get(bot.corpus_id)
    return store_mod.VectorStore(cfg.chroma_dir, corpus.collection).count()


def _delete_one_chunk(cfg, bot, vid: str) -> None:
    """Simulate a crash that left ``vid`` PARTIAL: drop a single one of its chunks
    straight from Chroma (bypassing the rebuild path)."""
    corpus = bot_service.corpus_store(cfg).get(bot.corpus_id)
    col = store_mod.VectorStore(cfg.chroma_dir, corpus.collection)._col
    got = col.get(include=["metadatas"])
    victim = next(cid for cid, m in zip(got["ids"], got["metadatas"])
                  if m and m.get("video_id") == vid)
    col.delete(ids=[victim])


def main() -> int:
    from ytrag.config import test_mode
    if test_mode():
        print("ERROR: YTRAG_TEST_MODE is set; this test needs the real embed pipeline.")
        return 1

    print("boot ok")
    cfg = load_config()
    bot_service.make_embedder = lambda *_a, **_k: RecordingEmbedder()  # type: ignore[assignment]
    store_mod.VectorStore.drop = _tracking_drop  # type: ignore[assignment]
    store_mod.VectorStore.add = _tracking_add  # type: ignore[assignment]

    bot = bot_service.create_bot(cfg, "u_test", "Inc Bot", "persona", "desc")
    source = bot_service.add_youtube_source(cfg, bot, "@chan", "Channel", "Author")

    # ---- Scenario 1: incremental FIRST build (empty collection) ----------
    _seed(cfg, bot, source, *V_OLD)
    _seed(cfg, bot, source, *V_NEW)
    EMBED_LOG.clear(); DROPPED.clear(); ADD_CALLS.clear()
    n1 = bot_service.rebuild_bot(cfg, bot)  # incremental (default)
    base_count = _collection_count(cfg, bot)
    check("incremental first build embeds both seed videos",
          _embedded_video_set() == {V_OLD[0], V_NEW[0]}, str(_embedded_video_set()))
    check("store.add is batched PER VIDEO (each add() call holds one video's chunks)",
          len(ADD_CALLS) == 2 and all(len(s) == 1 for s in ADD_CALLS), str(ADD_CALLS))
    check("incremental first build never called store.drop()", DROPPED == [], str(DROPPED))
    check("incremental first build indexed all chunks", n1 > 0 and base_count == n1,
          f"returned={n1} count={base_count}")
    check("collection now holds both videos",
          _collection_video_ids(cfg, bot) == {V_OLD[0], V_NEW[0]},
          str(_collection_video_ids(cfg, bot)))

    # ---- Scenario 2 (a): incremental ADD embeds ONLY the new video -------
    _seed(cfg, bot, source, *V_MID)  # a newly-published video appears
    EMBED_LOG.clear(); DROPPED.clear()
    bot_service.rebuild_bot(cfg, bot)  # incremental
    added = len(EMBED_LOG)
    new_count = _collection_count(cfg, bot)
    check("incremental add embeds ONLY the new video (not the existing ones)",
          _embedded_video_set() == {V_MID[0]}, str(_embedded_video_set()))
    check("incremental add never called store.drop()", DROPPED == [], str(DROPPED))
    check("incremental add PRESERVED existing chunks (count grew by exactly the new)",
          new_count == base_count + added and added > 0,
          f"before={base_count} after={new_count} added={added}")
    check("collection now holds all three videos",
          _collection_video_ids(cfg, bot) == {V_OLD[0], V_MID[0], V_NEW[0]},
          str(_collection_video_ids(cfg, bot)))

    # ---- Scenario 3 (b + c): FULL rebuild drops + re-embeds ALL, newest-first ----
    EMBED_LOG.clear(); DROPPED.clear()
    corpus = bot_service.corpus_store(cfg).get(bot.corpus_id)
    n3 = bot_service.rebuild_bot(cfg, bot, full=True)
    check("full rebuild called store.drop() on the collection",
          corpus.collection in DROPPED, str(DROPPED))
    check("full rebuild re-embedded EVERY video",
          _embedded_video_set() == {V_OLD[0], V_MID[0], V_NEW[0]}, str(_embedded_video_set()))
    check("full rebuild processed videos NEWEST-FIRST (by upload_date, not id)",
          _embed_video_order() == [V_NEW[0], V_MID[0], V_OLD[0]], str(_embed_video_order()))
    check("full rebuild chunk count unchanged (idempotent content)",
          n3 == new_count, f"full={n3} incremental={new_count}")

    # ---- Scenario 4 (d): cancelled INCREMENTAL rebuild must NOT drop -----
    _seed(cfg, bot, source, *V_EXTRA)  # a 4th new video to give the embed something to do
    EMBED_LOG.clear(); DROPPED.clear()
    ABORT["arm"] = True; ABORT["fire"] = False
    before_cancel = _collection_count(cfg, bot)
    r4 = bot_service.rebuild_bot(cfg, bot, should_abort=lambda: ABORT["fire"])
    after_cancel = _collection_count(cfg, bot)
    check("cancelled incremental rebuild returns 0 (aborted)", r4 == 0, f"returned={r4}")
    check("cancelled incremental rebuild did NOT drop the collection", DROPPED == [], str(DROPPED))
    check("cancelled incremental rebuild PRESERVED existing vectors",
          after_cancel == before_cancel and after_cancel > 0,
          f"before={before_cancel} after={after_cancel}")
    check("cancelled incremental rebuild did NOT add the new video",
          V_EXTRA[0] not in _collection_video_ids(cfg, bot),
          str(_collection_video_ids(cfg, bot)))

    # ---- Scenario 5 (d): cancelled FULL rebuild DOES drop ----------------
    EMBED_LOG.clear(); DROPPED.clear()
    ABORT["arm"] = True; ABORT["fire"] = False
    corpus = bot_service.corpus_store(cfg).get(bot.corpus_id)
    r5 = bot_service.rebuild_bot(cfg, bot, full=True, should_abort=lambda: ABORT["fire"])
    check("cancelled full rebuild returns 0 (aborted)", r5 == 0, f"returned={r5}")
    check("cancelled full rebuild DID drop the collection (no partial orphan)",
          corpus.collection in DROPPED, str(DROPPED))
    ABORT["arm"] = False; ABORT["fire"] = False

    # ---- Scenario 6 (F1): a PARTIAL video self-heals; a COMPLETE one is skipped --
    # Fresh bot so the partial-crash setup is isolated from the scenarios above.
    bot2 = bot_service.create_bot(cfg, "u_test", "Heal Bot", "persona", "desc")
    src2 = bot_service.add_youtube_source(cfg, bot2, "@chan2", "Channel2", "Author2")
    _seed(cfg, bot2, src2, *V_MULTI, reps=90)  # long transcript -> several chunks
    _seed(cfg, bot2, src2, *V_SOLO)            # single-chunk control
    expected_multi = _expected_chunks(cfg, bot2, src2, V_MULTI[0])
    bot_service.rebuild_bot(cfg, bot2)  # incremental first build -> both fully indexed
    check("multi-chunk video actually produced >1 chunk (setup sanity)",
          expected_multi >= 2, f"expected={expected_multi}")
    check("both videos fully indexed after first build",
          _chunk_counts(cfg, bot2).get(V_MULTI[0]) == expected_multi
          and _chunk_counts(cfg, bot2).get(V_SOLO[0]) == 1,
          str(_chunk_counts(cfg, bot2)))

    # Simulate a crash that left V_MULTI with one chunk missing (partial).
    _delete_one_chunk(cfg, bot2, V_MULTI[0])
    check("partial state: V_MULTI now has fewer than expected chunks",
          _chunk_counts(cfg, bot2).get(V_MULTI[0]) == expected_multi - 1,
          str(_chunk_counts(cfg, bot2)))

    EMBED_LOG.clear(); DROPPED.clear(); ADD_CALLS.clear()
    bot_service.rebuild_bot(cfg, bot2)  # incremental: should re-embed ONLY the partial
    check("partial video was RE-EMBEDDED (self-heal), complete one was NOT",
          _embedded_video_set() == {V_MULTI[0]}, str(_embedded_video_set()))
    check("re-embed rewrote the WHOLE video (idempotent upsert by id)",
          len(EMBED_LOG) == expected_multi, f"embedded={len(EMBED_LOG)} expected={expected_multi}")
    check("V_MULTI healed to full expected chunk count",
          _chunk_counts(cfg, bot2).get(V_MULTI[0]) == expected_multi,
          str(_chunk_counts(cfg, bot2)))
    check("complete V_SOLO untouched (still exactly 1 chunk, not re-embedded)",
          _chunk_counts(cfg, bot2).get(V_SOLO[0]) == 1 and V_SOLO[0] not in _embedded_video_set(),
          str(_chunk_counts(cfg, bot2)))
    check("healing incremental never called store.drop()", DROPPED == [], str(DROPPED))

    # ---- Scenario 7: a SHORTENED transcript (present > expected) reindexes once --
    bot3 = bot_service.create_bot(cfg, "u_test", "Shrink Bot", "persona", "desc")
    src3 = bot_service.add_youtube_source(cfg, bot3, "@chan3", "Channel3", "Author3")
    _seed(cfg, bot3, src3, *V_SHRINK, reps=90)  # long transcript -> several chunks
    expected_big = _expected_chunks(cfg, bot3, src3, V_SHRINK[0])
    bot_service.rebuild_bot(cfg, bot3)  # fully index the big version
    check("shrink setup: big version fully indexed (>1 chunk)",
          expected_big >= 2 and _chunk_counts(cfg, bot3).get(V_SHRINK[0]) == expected_big,
          f"expected_big={expected_big} counts={_chunk_counts(cfg, bot3)}")

    # The channel owner re-uploaded a MUCH shorter transcript (no full rebuild).
    _seed(cfg, bot3, src3, *V_SHRINK, reps=1)  # overwrite on disk -> fewer chunks
    expected_small = _expected_chunks(cfg, bot3, src3, V_SHRINK[0])
    check("shrink setup: present now EXCEEDS expected (orphan tail on disk-shrink)",
          expected_small < expected_big
          and _chunk_counts(cfg, bot3).get(V_SHRINK[0]) == expected_big,
          f"small={expected_small} big={expected_big} counts={_chunk_counts(cfg, bot3)}")

    EMBED_LOG.clear(); DROPPED.clear()
    bot_service.rebuild_bot(cfg, bot3)  # incremental: reindex to the smaller set
    check("shortened video reindexed to the NEW smaller chunk set",
          _embedded_video_set() == {V_SHRINK[0]} and len(EMBED_LOG) == expected_small,
          f"embedded={_embedded_video_set()} n={len(EMBED_LOG)} expected={expected_small}")
    check("orphan tail removed (present == expected after reindex)",
          _chunk_counts(cfg, bot3).get(V_SHRINK[0]) == expected_small,
          str(_chunk_counts(cfg, bot3)))
    check("shrink reindex never called store.drop() (scoped per-video delete)",
          DROPPED == [], str(DROPPED))

    EMBED_LOG.clear()
    bot_service.rebuild_bot(cfg, bot3)  # next incremental: must be a no-op (no loop)
    check("no infinite loop: shortened video is SKIPPED on the next incremental",
          V_SHRINK[0] not in _embedded_video_set()
          and _chunk_counts(cfg, bot3).get(V_SHRINK[0]) == expected_small,
          f"embedded={_embedded_video_set()} counts={_chunk_counts(cfg, bot3)}")

    store_mod.VectorStore.add = _orig_add  # type: ignore[assignment]
    store_mod.VectorStore.drop = _orig_drop  # type: ignore[assignment]
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)
    print(f"\n==== {passed}/{total} checks passed ====")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
