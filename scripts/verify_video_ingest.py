"""Offline verification for single-VIDEO source support.

A video is a YouTube source of ``kind='video'`` (type stays ``youtube``), keyed
off its case-sensitive 11-char video id. Monkeypatches yt-dlp so no network is
touched, then asserts:
  1. normalize_video_key extracts the id from watch?v=, youtu.be/, /shorts/<id>,
     and /embed/<id> → "video/<ID>" (case preserved), and REJECTS non-video
     input (channel/playlist/empty) → "".
  2. _is_video_key / _video_url recognize the key and build the canonical watch
     URL.
  3. extract_channel('video/<id>') does a SINGLE extraction of the watch URL and
     does NOT touch /videos or /shorts (no channel merge); the lone video (whose
     info dict has no ``entries``) is wrapped as ONE {id,title,url} entry and the
     meta title is the VIDEO title (not the uploader/channel).
  4. an extractor error on a video RAISES (no fallback tab).
  5. a raw watch URL still passes through as a single video.
  6. add-youtube kind='video' persists BotSource.kind='video' + key 'video/<id>',
     echoes kind back, and defaults sync_freq to 'off' (a single video needs no
     schedule); an explicit sync_freq is respected; a non-video URL → 400; an
     unknown kind → 400.

Run: .venv/Scripts/python.exe scripts/verify_video_ingest.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

os.environ.setdefault("YTRAG_DEV_LOGIN", "1")
os.environ.pop("YTRAG_ENV", None)
os.environ.setdefault("YTRAG_DATA_DIR", tempfile.mkdtemp(prefix="ytrag_video_verify_"))

import ytrag.ingest as ingest  # noqa: E402
from ytrag.sources import normalize_video_key  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []

_ID = "dQw4w9WgXcQ"  # a canonical 11-char video id


def check(name: str, passed: bool, detail: str = "") -> None:
    RESULTS.append((name, passed, bool(detail) and str(detail)))


class FakeYDL:
    """Stand-in for YoutubeDL returning canned info keyed by a URL substring, and
    RECORDING every URL it was asked to extract (to prove no /videos+/shorts)."""

    responses: dict = {}
    seen: list[str] = []

    def __init__(self, opts):
        self.opts = opts

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download=False):
        FakeYDL.seen.append(url)
        for needle, value in self.responses.items():
            if needle in url:
                if isinstance(value, Exception):
                    raise value
                return value
        return None


def _video_info(vid, title):
    """A single-video info dict as yt-dlp returns it: NO ``entries`` key."""
    return {
        "id": vid,
        "title": title,
        "channel": "Some Channel Name",  # must be IGNORED for the video's meta title
        "uploader": "Some Uploader",
        "webpage_url": f"https://www.youtube.com/watch?v={vid}",
        "thumbnails": [{"id": "thumb", "url": f"http://img/{vid}.jpg", "width": 1280}],
    }


# --- 1: normalize_video_key -------------------------------------------------
check("watch?v= -> video/<id>",
      normalize_video_key(f"https://www.youtube.com/watch?v={_ID}") == f"video/{_ID}",
      normalize_video_key(f"https://www.youtube.com/watch?v={_ID}"))
check("watch?v= with extra params -> video/<id>",
      normalize_video_key(f"https://www.youtube.com/watch?v={_ID}&t=42s&list=PLx") == f"video/{_ID}",
      normalize_video_key(f"https://www.youtube.com/watch?v={_ID}&t=42s&list=PLx"))
check("youtu.be/<id> -> video/<id>",
      normalize_video_key(f"https://youtu.be/{_ID}") == f"video/{_ID}",
      normalize_video_key(f"https://youtu.be/{_ID}"))
check("/shorts/<id> -> video/<id>",
      normalize_video_key(f"https://www.youtube.com/shorts/{_ID}") == f"video/{_ID}",
      normalize_video_key(f"https://www.youtube.com/shorts/{_ID}"))
check("/embed/<id> -> video/<id>",
      normalize_video_key(f"https://www.youtube.com/embed/{_ID}") == f"video/{_ID}",
      normalize_video_key(f"https://www.youtube.com/embed/{_ID}"))
check("video id case is PRESERVED (not lower-cased)",
      normalize_video_key("https://youtu.be/AbCdEfGhIjK") == "video/AbCdEfGhIjK",
      normalize_video_key("https://youtu.be/AbCdEfGhIjK"))
check("channel URL -> '' (not a video)", normalize_video_key("https://www.youtube.com/@Igor") == "",
      normalize_video_key("https://www.youtube.com/@Igor"))
check("bare @handle -> ''", normalize_video_key("@Igor") == "")
check("playlist URL -> ''", normalize_video_key("https://www.youtube.com/playlist?list=PLabcдеf123") == "")
check("channel/UC… URL -> ''", normalize_video_key("https://www.youtube.com/channel/UCabcdefg") == "")
check("empty input -> ''", normalize_video_key("") == "")
# Never collides with channel/playlist namespaces.
check("video key namespaced apart from playlist/channel",
      normalize_video_key(f"https://youtu.be/{_ID}").startswith("video/"))

# --- 2: _is_video_key / _video_url ------------------------------------------
check("_is_video_key('video/<id>')", ingest._is_video_key(f"video/{_ID}"))
check("_is_video_key('@Igor') is False", not ingest._is_video_key("@Igor"))
check("_is_video_key('playlist/PLabc') is False", not ingest._is_video_key("playlist/PLabc"))
check("_video_url builds canonical watch URL",
      ingest._video_url(f"video/{_ID}") == f"https://www.youtube.com/watch?v={_ID}",
      ingest._video_url(f"video/{_ID}"))

# --- 3: extract_channel('video/<id>'): single watch extraction, no channel tabs
FakeYDL.seen = []
FakeYDL.responses = {f"watch?v={_ID}": _video_info(_ID, "The 3-Hour Sourdough Deep Dive")}
ingest.YoutubeDL = FakeYDL  # type: ignore[assignment]
meta, videos = ingest.extract_channel(f"video/{_ID}")
check("video enumerates exactly ONE entry", [v["id"] for v in videos] == [_ID],
      f"got {[v['id'] for v in videos]}")
check("video entry title is the video title",
      videos and videos[0]["title"] == "The 3-Hour Sourdough Deep Dive",
      videos[0]["title"] if videos else "<none>")
check("video meta title is the VIDEO title (not channel)",
      meta.get("title") == "The 3-Hour Sourdough Deep Dive", f"got {meta.get('title')!r}")
check("video avatar picked from thumbnails",
      meta.get("avatar") == f"http://img/{_ID}.jpg", f"got {meta.get('avatar')!r}")
check("video did NOT hit /videos tab", not any("/videos" in u for u in FakeYDL.seen),
      f"urls seen: {FakeYDL.seen}")
check("video did NOT hit /shorts tab", not any("/shorts" in u for u in FakeYDL.seen),
      f"urls seen: {FakeYDL.seen}")
check("video hit exactly one canonical watch page",
      FakeYDL.seen == [f"https://www.youtube.com/watch?v={_ID}"], f"urls seen: {FakeYDL.seen}")

# list_video_ids delegates to extract_channel → stays a single flat entry.
FakeYDL.seen = []
ids = [v["id"] for v in ingest.list_video_ids(f"video/{_ID}")]
check("list_video_ids(video key) -> single entry", ids == [_ID], f"got {ids}")

# --- 4: extractor error on a video RAISES (no fallback) ---------------------
def _raises(responses, ref):
    FakeYDL.responses = responses
    FakeYDL.seen = []
    ingest.YoutubeDL = FakeYDL  # type: ignore[assignment]
    try:
        ingest.extract_channel(ref)
        return False
    except Exception:
        return True


check("video extractor error RAISES",
      _raises({"watch?v=boomboom123": RuntimeError("network")}, "video/boomboom123"))

# --- 5: a raw watch URL still passes through as a single video ---------------
FakeYDL.seen = []
FakeYDL.responses = {"watch?v=solo1234567": _video_info("solo1234567", "Solo Talk")}
ingest.YoutubeDL = FakeYDL  # type: ignore[assignment]
_meta, videos = ingest.extract_channel("https://www.youtube.com/watch?v=solo1234567")
check("raw watch URL -> single video entry", [v["id"] for v in videos] == ["solo1234567"],
      f"got {[v['id'] for v in videos]}")
check("raw watch URL did NOT hit channel tabs",
      not any("/videos" in u or "/shorts" in u for u in FakeYDL.seen), f"urls seen: {FakeYDL.seen}")

# --- 6: add-youtube endpoint persists kind='video' --------------------------
from fastapi.testclient import TestClient  # noqa: E402

from ytrag.web.app import app  # noqa: E402
from ytrag.config import load_config  # noqa: E402
from ytrag.bots import TYPE_YOUTUBE  # noqa: E402
from ytrag import bot_service  # noqa: E402

client = TestClient(app, follow_redirects=False)
client.get("/auth/dev")  # dev session cookie
cfg = load_config()

bot = client.post("/api/bots", json={"name": "Video Bot"}).json()["bot"]
bot_id = bot["id"]

# kind='video' with a real watch URL → persists kind + key, echoes kind, sync off.
r = client.post(f"/api/bots/{bot_id}/sources/youtube", json={
    "channel": f"https://www.youtube.com/watch?v={_ID}",
    "author": "Alex", "consent": True, "kind": "video",
})
check("add video -> 200", r.status_code == 200, f"got {r.status_code}: {r.text[:160]}")
check("add video echoes kind='video'", r.json().get("kind") == "video", r.text[:160])
check("video sync_freq defaults to 'off'", r.json().get("sync_freq") == "off", r.text[:160])
vid_source_id = r.json().get("source_id")
persisted = bot_service.bot_store(cfg).get(bot_id)
v_src = next((s for s in persisted.sources if s.id == vid_source_id), None)
check("video BotSource.kind persisted", v_src is not None and v_src.kind == "video",
      getattr(v_src, "kind", None))
check("video BotSource keeps type=youtube", v_src is not None and v_src.type == TYPE_YOUTUBE)
check("video key is video/<id>", v_src is not None and v_src.key == f"video/{_ID}",
      getattr(v_src, "key", None))
check("video BotSource sync_freq persisted 'off'", v_src is not None and v_src.sync_freq == "off",
      getattr(v_src, "sync_freq", None))

# An explicit sync_freq on a video is respected (not overridden by the 'off' default).
r = client.post(f"/api/bots/{bot_id}/sources/youtube", json={
    "channel": "https://youtu.be/AbCdEfGhIjK",
    "author": "Alex", "consent": True, "kind": "video", "sync_freq": "weekly",
})
check("explicit video sync_freq respected", r.status_code == 200 and r.json().get("sync_freq") == "weekly",
      f"got {r.status_code}: {r.text[:160]}")

# a non-video URL in video mode → 400.
r = client.post(f"/api/bots/{bot_id}/sources/youtube", json={
    "channel": "https://www.youtube.com/@notavideo",
    "author": "Alex", "consent": True, "kind": "video",
})
check("video kind + non-video URL -> 400", r.status_code == 400, f"got {r.status_code}")

# unknown kind → 400 (mentions the three valid kinds).
r = client.post(f"/api/bots/{bot_id}/sources/youtube", json={
    "channel": "@x", "author": "Alex", "consent": True, "kind": "bogus",
})
check("unknown kind -> 400", r.status_code == 400, f"got {r.status_code}")
check("unknown-kind 400 lists 'video'", "video" in (r.json().get("error", "").lower()), r.text[:160])

# --- report -----------------------------------------------------------------
print()
all_ok = True
for name, passed, detail in RESULTS:
    status = "PASS" if passed else "FAIL"
    all_ok = all_ok and passed
    line = f"[{status}] {name}"
    if detail and not passed:
        line += f"  -> {detail}"
    print(line)
print()
print("ALL PASS" if all_ok else "SOME FAILED")
sys.exit(0 if all_ok else 1)
