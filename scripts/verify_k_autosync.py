"""Phase K verification — per-channel auto-sync.

Isolated temp data+chroma dir, NO network (channel listing + the FIFO enqueue are
monkeypatched), dev-login. Proves:

  * BotSource sync_freq/last_sync_at defaults + backward-compat loading of old
    source records that predate Phase K
  * a new YouTube source inherits the user's default sync frequency at add time
  * the sync_freq update endpoint (auth + ownership) persists a per-source freq
  * the incremental diff enqueues ONLY new video keys through the single FIFO
    queue, stamps last_sync_at, and returns the new-video count
  * the "Sync now" endpoint returns the new count and enqueues one job
  * the scheduler's interval-elapsed logic (off / never-synced / each freq) and a
    tick that triggers only due sources
  * user default sync_freq round-trips through /api/me/settings (+ 400 on garbage)
"""
from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

_tmp = Path(tempfile.mkdtemp(prefix="k_"))
os.environ["YTRAG_ENV"] = "development"
os.environ["YTRAG_DEV_LOGIN"] = "1"
os.environ["YTRAG_DATA_DIR"] = str(_tmp / "data")
os.environ["YTRAG_CHROMA_DIR"] = str(_tmp / "chroma")
os.environ["YTRAG_SESSION_SECRET"] = "test-secret-k"

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from starlette.testclient import TestClient  # noqa: E402

from ytrag import autosync, bot_service  # noqa: E402
from ytrag import ingest as ingest_mod  # noqa: E402
from ytrag import ingest_queue  # noqa: E402
from ytrag.bots import BotSource, _bot_from_rec  # noqa: E402
from ytrag.config import load_config  # noqa: E402
from ytrag.ingest_jobs import JobStore  # noqa: E402
from ytrag.user_settings import UserSettingsStore  # noqa: E402
from ytrag.web.app import app  # noqa: E402

_results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


# ---- no-network stubs -----------------------------------------------------
# Record every FIFO enqueue instead of starting the real download worker.
_enqueued: list[dict] = []


def _fake_enqueue(cfg, bot, source, **kw):
    _enqueued.append({"bot_id": bot.id, "source_id": source.id, **kw})
    return "job-" + source.id


# The channel listing the diff runs against — swapped per test.
_channel_videos: list[dict] = []


def _fake_list_video_ids(channel, proxy=None, include_shorts=True):
    return list(_channel_videos)


def _write_transcripts(cfg, bot, source, ids) -> None:
    tdir = bot_service.source_transcripts_dir(cfg, bot.corpus_id, source.id)
    tdir.mkdir(parents=True, exist_ok=True)
    for vid in ids:
        (tdir / f"{vid}.json").write_text('{"id": "%s", "text": "x"}' % vid,
                                          encoding="utf-8")


def main() -> int:
    print("boot ok")
    # Install no-network stubs BEFORE any sync path runs.
    ingest_queue.manager.enqueue = _fake_enqueue
    ingest_mod.list_video_ids = _fake_list_video_ids

    cfg = load_config()

    # ---- 1) dataclass defaults + backward-compat ----------------------
    src = BotSource(id="s1", type="youtube", key="@c", label="c", author="a")
    check("BotSource.sync_freq defaults to '' (inherit)", src.sync_freq == "")
    check("BotSource.last_sync_at defaults to ''", src.last_sync_at == "")
    check("BotSource.first_indexed_at defaults to ''", src.first_indexed_at == "")
    # An old bot record with a source lacking the Phase-K fields still loads.
    legacy = _bot_from_rec({
        "id": "b0", "owner_id": "u", "name": "n", "persona": "", "description": "",
        "corpus_id": "c0", "created_at": "2026-01-01T00:00:00Z",
        "sources": [{"id": "s0", "type": "youtube", "key": "@old",
                     "label": "old", "author": "a"}],
    })
    check("legacy source record loads (no sync_freq key)",
          legacy.sources[0].sync_freq == "" and legacy.sources[0].last_sync_at == "")

    # ---- 2) interval_elapsed / effective_freq (scheduler logic) --------
    now = datetime(2026, 8, 10, 12, 0, 0, tzinfo=timezone.utc)
    check("interval: off is never due", autosync.interval_elapsed("off", "", now) is False)
    check("interval: unknown is never due", autosync.interval_elapsed("bogus", "", now) is False)
    check("interval: never-synced daily is due", autosync.interval_elapsed("daily", "", now) is True)
    day_ago = (now - timedelta(hours=25)).isoformat()
    hour_ago = (now - timedelta(hours=1)).isoformat()
    check("interval: daily due after 25h", autosync.interval_elapsed("daily", day_ago, now) is True)
    check("interval: daily NOT due after 1h", autosync.interval_elapsed("daily", hour_ago, now) is False)
    six_days = (now - timedelta(days=6)).isoformat()
    eight_days = (now - timedelta(days=8)).isoformat()
    check("interval: weekly NOT due after 6d", autosync.interval_elapsed("weekly", six_days, now) is False)
    check("interval: weekly due after 8d", autosync.interval_elapsed("weekly", eight_days, now) is True)
    check("effective_freq: source overrides default",
          autosync.effective_freq("daily", "weekly") == "daily")
    check("effective_freq: blank source inherits default",
          autosync.effective_freq("", "weekly") == "weekly")
    check("effective_freq: blank+blank -> off",
          autosync.effective_freq("", "") == "off")

    # ---- 2b) source_is_due: auto-sync gated on initial import (BUG-020) ----
    stamped_now = now.isoformat()
    old_stamp = (now - timedelta(days=8)).isoformat()
    check("due: off never due", autosync.source_is_due("off", old_stamp, "", now) is False)
    check("due: no first_indexed -> NOT due (import never completed)",
          autosync.source_is_due("weekly", "", "", now) is False)
    check("due: legacy channel synced 8d ago, no first_indexed -> due (no-backfill compat)",
          autosync.source_is_due("weekly", "", old_stamp, now) is True)
    check("due: legacy channel synced 1h ago, no first_indexed -> NOT due",
          autosync.source_is_due("weekly", "", hour_ago, now) is False)
    check("due: just-completed import (interval not elapsed) -> NOT due",
          autosync.source_is_due("weekly", stamped_now, "", now) is False)
    check("due: completed 8d ago, never synced -> due (baseline=first_indexed_at)",
          autosync.source_is_due("weekly", old_stamp, "", now) is True)
    check("due: completed long ago but synced 1h ago -> NOT due (baseline=last_sync)",
          autosync.source_is_due("weekly", old_stamp, hour_ago, now) is False)
    check("due: completed long ago, last sync 8d ago -> due",
          autosync.source_is_due("weekly", old_stamp, old_stamp, now) is True)

    # ---- 3) diff enqueues ONLY new keys -------------------------------
    current = [{"id": "vidA"}, {"id": "vidB"}, {"id": "vidC"}]
    new = autosync.diff_new_video_ids(current, {"vidA", "vidB"})
    check("diff: returns only new ids", new == ["vidC"], str(new))
    check("diff: empty when all ingested",
          autosync.diff_new_video_ids(current, {"vidA", "vidB", "vidC"}) == [])

    # incremental_sync end-to-end (no network): 2 of 3 already on disk -> 1 new.
    bot = bot_service.create_bot(cfg, owner_id="u_dev_test", name="B", persona="",
                                 description="", language="")
    ysrc = bot_service.add_youtube_source(cfg, bot, "@chan", "@chan", "Author")
    _write_transcripts(cfg, bot, ysrc, ["vidA", "vidB"])
    global _channel_videos
    _channel_videos = current
    _enqueued.clear()
    n = autosync.incremental_sync(cfg, bot, ysrc, user_id=bot.owner_id)
    check("incremental_sync returns new count (1)", n == 1, f"n={n}")
    check("incremental_sync enqueued exactly one job", len(_enqueued) == 1, str(_enqueued))
    check("incremental_sync enqueued THIS source",
          _enqueued and _enqueued[0]["source_id"] == ysrc.id)
    reloaded = bot_service.bot_store(cfg).get(bot.id)
    stamped = reloaded.sources[0].last_sync_at
    check("incremental_sync stamped last_sync_at", bool(stamped), stamped)

    # Nothing new -> no enqueue, count 0, but still re-stamps.
    _write_transcripts(cfg, bot, ysrc, ["vidC"])  # now all 3 present
    _enqueued.clear()
    n2 = autosync.incremental_sync(cfg, bot, ysrc, user_id=bot.owner_id)
    check("incremental_sync: 0 new -> no enqueue", n2 == 0 and not _enqueued, f"n2={n2}")

    # ---- 4) scheduler tick only triggers due sources ------------------
    # A weekly source whose initial import COMPLETED >1 interval ago is due; an
    # 'off' source is not; and a weekly source whose import never completed
    # (no first_indexed_at) must NOT be auto-synced (BUG-020).
    bot2 = bot_service.create_bot(cfg, owner_id="u_dev_test", name="B2", persona="",
                                  description="", language="")
    due = bot_service.add_youtube_source(cfg, bot2, "@due", "@due", "A")
    bot_service.bot_store(cfg).update_source(
        bot2.id, due.id, sync_freq="weekly",
        first_indexed_at=(now - timedelta(days=8)).isoformat())
    off = bot_service.add_youtube_source(cfg, bot2, "@off", "@off", "A")
    bot_service.bot_store(cfg).update_source(bot2.id, off.id, sync_freq="off")
    # Weekly, but its initial import never finished -> never auto-synced.
    nocomplete = bot_service.add_youtube_source(cfg, bot2, "@nc", "@nc", "A")
    bot_service.bot_store(cfg).update_source(bot2.id, nocomplete.id, sync_freq="weekly")
    _channel_videos = [{"id": "newvid"}]
    _enqueued.clear()
    sched = autosync.AutoSyncScheduler(load_config)
    triggered = sched.tick(now=now + timedelta(days=1))
    synced_ids = {e["source_id"] for e in _enqueued}
    check("tick triggered the due weekly source", due.id in synced_ids, str(synced_ids))
    check("tick skipped the off source", off.id not in synced_ids)
    check("tick skipped never-completed source (BUG-020)",
          nocomplete.id not in synced_ids, str(synced_ids))
    check("tick returned a positive trigger count", triggered >= 1, str(triggered))

    # ---- app / dev login ---------------------------------------------
    client = TestClient(app)
    client.get("/auth/dev", follow_redirects=False)

    # ---- 5) user default round-trips via /api/me/settings -------------
    r = client.post("/api/me/settings", json={"sync_freq": "weekly"})
    check("POST settings sync_freq=weekly ok", r.status_code == 200, r.text[:120])
    g = client.get("/api/me/settings").json()
    check("GET settings reflects sync_freq=weekly", g.get("sync_freq") == "weekly", str(g.get("sync_freq")))
    bad = client.post("/api/me/settings", json={"sync_freq": "hourly"})
    check("POST settings rejects unknown freq (400)", bad.status_code == 400, str(bad.status_code))

    # ---- 6) new YouTube source inherits the user default --------------
    b = client.post("/api/bots", json={"name": "InheritBot", "description": "d"}).json()["bot"]
    add = client.post(
        f"/api/bots/{b['id']}/sources/youtube",
        json={"channel": "@inherit", "author": "A", "consent": True},
    )
    check("add youtube ok", add.status_code == 200, add.text[:160])
    check("new source inherits user default (weekly)",
          add.json().get("sync_freq") == "weekly", add.json().get("sync_freq"))

    # explicit choice at add time wins over the default
    add2 = client.post(
        f"/api/bots/{b['id']}/sources/youtube",
        json={"channel": "@explicit", "author": "A", "consent": True, "sync_freq": "daily"},
    )
    check("explicit sync_freq at add wins", add2.json().get("sync_freq") == "daily",
          add2.json().get("sync_freq"))

    # ---- 7) sync_freq update endpoint --------------------------------
    sid = add.json()["source_id"]
    upd = client.post(f"/api/bots/{b['id']}/sources/{sid}/sync-freq", json={"sync_freq": "monthly"})
    check("sync-freq update ok", upd.status_code == 200 and upd.json().get("sync_freq") == "monthly",
          upd.text[:120])
    # blank resets to inherit ('')
    reset = client.post(f"/api/bots/{b['id']}/sources/{sid}/sync-freq", json={"sync_freq": ""})
    check("sync-freq blank resets to inherit", reset.json().get("sync_freq") == "", reset.text[:120])
    badf = client.post(f"/api/bots/{b['id']}/sources/{sid}/sync-freq", json={"sync_freq": "yearly"})
    check("sync-freq rejects unknown value (400)", badf.status_code == 400, str(badf.status_code))

    # ---- 8) Sync now endpoint ----------------------------------------
    _channel_videos = [{"id": "s1"}, {"id": "s2"}]  # fresh source -> both new
    _enqueued.clear()
    JobStore(cfg.data_dir)  # ensure store dir exists
    sync = client.post(f"/api/bots/{b['id']}/sources/{sid}/sync")
    check("sync-now ok", sync.status_code == 200, sync.text[:160])
    check("sync-now reports new count (2)", sync.json().get("new") == 2, sync.text[:160])
    check("sync-now enqueued one job", len(_enqueued) == 1, str(_enqueued))

    # ---- summary ------------------------------------------------------
    passed = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print(f"\n{passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
