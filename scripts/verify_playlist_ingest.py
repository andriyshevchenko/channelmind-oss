"""Offline verification for YouTube PLAYLIST source support.

A playlist is a YouTube source of ``kind='playlist'`` (type stays ``youtube``),
keyed off its case-sensitive ``list=`` id. Monkeypatches yt-dlp so no network is
touched, then asserts:
  1. normalize_playlist_key extracts list= → "playlist/<PLID>" (case preserved),
     accepts a bare PL… id, and REJECTS personal/auto lists (LL/WL/RD/UL) and
     non-playlist input.
  2. _is_playlist_ref / _playlist_url recognize both the key and raw URLs.
  3. extract_channel('playlist/PLabc') does a SINGLE flat extraction of the
     playlist page and does NOT touch /videos or /shorts (no channel merge).
  4. a watch?v=…&list=… URL is treated as a playlist (branch runs BEFORE the
     single-video check).
  5. playlist meta title comes from info["title"]; list_video_ids stays flat.
  6. an extractor error on a playlist RAISES (no fallback tab).
  7. add-youtube kind='playlist' persists BotSource.kind and echoes kind back;
     kind='channel' (and the default) stays a channel; a mix id is 400-rejected.

Run: .venv/Scripts/python.exe scripts/verify_playlist_ingest.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

os.environ.setdefault("YTRAG_DEV_LOGIN", "1")
os.environ.pop("YTRAG_ENV", None)
os.environ.setdefault("YTRAG_DATA_DIR", tempfile.mkdtemp(prefix="ytrag_playlist_verify_"))

import ytrag.ingest as ingest  # noqa: E402
from ytrag.sources import normalize_playlist_key  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


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


def _pl_info(title, entries):
    return {
        "title": title,
        "channel": "Some Channel Name",  # must be IGNORED for a playlist title
        "thumbnails": [{"id": "thumb", "url": f"http://img/{title}.jpg", "width": 640}],
        "entries": entries,
    }


def _entry(vid, title):
    return {"id": vid, "title": title, "url": f"https://www.youtube.com/watch?v={vid}"}


# --- 1: normalize_playlist_key ----------------------------------------------
check("list= url -> playlist/PLabc",
      normalize_playlist_key("https://www.youtube.com/playlist?list=PLabc") == "playlist/PLabc",
      normalize_playlist_key("https://www.youtube.com/playlist?list=PLabc"))
check("watch?v=…&list=… -> playlist/<id>",
      normalize_playlist_key("https://www.youtube.com/watch?v=x&list=PLxyz123") == "playlist/PLxyz123",
      normalize_playlist_key("https://www.youtube.com/watch?v=x&list=PLxyz123"))
check("playlist id case is PRESERVED (not lower-cased)",
      normalize_playlist_key("https://youtube.com/playlist?list=PLAbCdEf") == "playlist/PLAbCdEf",
      normalize_playlist_key("https://youtube.com/playlist?list=PLAbCdEf"))
check("bare PL… id accepted", normalize_playlist_key("PLdeadbeef") == "playlist/PLdeadbeef",
      normalize_playlist_key("PLdeadbeef"))
check("Liked-list LL… rejected -> ''", normalize_playlist_key("https://youtube.com/playlist?list=LLsomething") == "",
      normalize_playlist_key("https://youtube.com/playlist?list=LLsomething"))
check("Watch-Later WL rejected -> ''", normalize_playlist_key("https://youtube.com/playlist?list=WLxyz") == "")
check("Radio/mix RD rejected -> ''", normalize_playlist_key("https://youtube.com/watch?v=a&list=RDabcdef") == "")
check("Uploads UL rejected -> ''", normalize_playlist_key("https://youtube.com/playlist?list=ULabcdef") == "")
check("no list= and not a bare id -> ''", normalize_playlist_key("https://youtube.com/@Igor") == "")
check("empty input -> ''", normalize_playlist_key("") == "")

# --- 2: _is_playlist_ref / _playlist_url ------------------------------------
check("_is_playlist_ref('playlist/PLabc')", ingest._is_playlist_ref("playlist/PLabc"))
check("_is_playlist_ref(url with list=)", ingest._is_playlist_ref("https://youtube.com/playlist?list=PLabc"))
check("_is_playlist_ref(watch?v&list=)", ingest._is_playlist_ref("https://youtube.com/watch?v=z&list=PLq"))
check("_is_playlist_ref('@Igor') is False", not ingest._is_playlist_ref("@Igor"))
check("_playlist_url from key",
      ingest._playlist_url("playlist/PLabc") == "https://www.youtube.com/playlist?list=PLabc",
      ingest._playlist_url("playlist/PLabc"))
check("_playlist_url from watch url drops video, keeps list",
      ingest._playlist_url("https://youtube.com/watch?v=vid&list=PLabc") == "https://www.youtube.com/playlist?list=PLabc",
      ingest._playlist_url("https://youtube.com/watch?v=vid&list=PLabc"))

# --- 3 + 5: extract_channel on a playlist key: single flat page, no channel tabs
FakeYDL.seen = []
FakeYDL.responses = {"list=PLabc": _pl_info("Knife Skills — Full Course",
                                            [_entry("a", "Lesson A"), _entry("b", "Lesson B")])}
ingest.YoutubeDL = FakeYDL  # type: ignore[assignment]
meta, videos = ingest.extract_channel("playlist/PLabc")
check("playlist enumerates its entries flat", [v["id"] for v in videos] == ["a", "b"],
      f"got {[v['id'] for v in videos]}")
check("playlist title from info['title'] (not channel)",
      meta.get("title") == "Knife Skills — Full Course", f"got {meta.get('title')!r}")
check("playlist avatar picked from thumbnails",
      meta.get("avatar") == "http://img/Knife Skills — Full Course.jpg", f"got {meta.get('avatar')!r}")
check("playlist did NOT hit /videos tab", not any("/videos" in u for u in FakeYDL.seen),
      f"urls seen: {FakeYDL.seen}")
check("playlist did NOT hit /shorts tab", not any("/shorts" in u for u in FakeYDL.seen),
      f"urls seen: {FakeYDL.seen}")
check("playlist hit exactly one playlist page", FakeYDL.seen == ["https://www.youtube.com/playlist?list=PLabc"],
      f"urls seen: {FakeYDL.seen}")

# list_video_ids delegates to extract_channel → stays flat for a playlist key.
FakeYDL.seen = []
ids = [v["id"] for v in ingest.list_video_ids("playlist/PLabc")]
check("list_video_ids(playlist key) flat", ids == ["a", "b"], f"got {ids}")

# --- 4: watch?v=…&list=… routes through the playlist branch, not single-video -
FakeYDL.seen = []
FakeYDL.responses = {"list=PLwatch": _pl_info("From Watch URL", [_entry("w", "W")])}
ingest.YoutubeDL = FakeYDL  # type: ignore[assignment]
_meta, videos = ingest.extract_channel("https://www.youtube.com/watch?v=single&list=PLwatch")
check("watch?v&list= treated as playlist (whole list, not the one video)",
      [v["id"] for v in videos] == ["w"] and FakeYDL.seen == ["https://www.youtube.com/playlist?list=PLwatch"],
      f"ids={[v['id'] for v in videos]} urls={FakeYDL.seen}")

# --- 6: extractor error on a playlist RAISES (no fallback) -------------------
def _raises(responses, ref):
    FakeYDL.responses = responses
    FakeYDL.seen = []
    ingest.YoutubeDL = FakeYDL  # type: ignore[assignment]
    try:
        ingest.extract_channel(ref)
        return False
    except Exception:
        return True


check("playlist extractor error RAISES",
      _raises({"list=PLboom": RuntimeError("proxy down")}, "playlist/PLboom"))

# --- 7: add-youtube endpoint persists kind ----------------------------------
from fastapi.testclient import TestClient  # noqa: E402

from ytrag.web.app import app  # noqa: E402
from ytrag.config import load_config  # noqa: E402
from ytrag.bots import TYPE_YOUTUBE  # noqa: E402
from ytrag import bot_service  # noqa: E402

client = TestClient(app, follow_redirects=False)
client.get("/auth/dev")  # dev session cookie
me = client.get("/api/me").json()["user"]
cfg = load_config()

bot = client.post("/api/bots", json={"name": "Playlist Bot"}).json()["bot"]
bot_id = bot["id"]

# kind='playlist' with a real list= URL → persists kind + echoes it back.
r = client.post(f"/api/bots/{bot_id}/sources/youtube", json={
    "channel": "https://www.youtube.com/playlist?list=PLverify123",
    "author": "Alex", "consent": True, "kind": "playlist",
})
check("add playlist -> 200", r.status_code == 200, f"got {r.status_code}: {r.text[:160]}")
check("add playlist echoes kind='playlist'", r.json().get("kind") == "playlist", r.text[:160])
pl_source_id = r.json().get("source_id")
persisted = bot_service.bot_store(cfg).get(bot_id)
pl_src = next((s for s in persisted.sources if s.id == pl_source_id), None)
check("playlist BotSource.kind persisted", pl_src is not None and pl_src.kind == "playlist",
      getattr(pl_src, "kind", None))
check("playlist BotSource keeps type=youtube", pl_src is not None and pl_src.type == TYPE_YOUTUBE)
check("playlist key is playlist/PLverify123", pl_src is not None and pl_src.key == "playlist/PLverify123",
      getattr(pl_src, "key", None))

# kind='channel' (explicit) → channel key, kind channel.
r = client.post(f"/api/bots/{bot_id}/sources/youtube", json={
    "channel": "@channelverify", "author": "Alex", "consent": True, "kind": "channel",
})
check("add channel -> 200", r.status_code == 200, f"got {r.status_code}: {r.text[:160]}")
check("add channel echoes kind='channel'", r.json().get("kind") == "channel", r.text[:160])
ch_src = next((s for s in bot_service.bot_store(cfg).get(bot_id).sources
               if s.id == r.json().get("source_id")), None)
check("channel BotSource.kind='channel'", ch_src is not None and ch_src.kind == "channel",
      getattr(ch_src, "kind", None))

# default (no kind) → channel (back-compat with the pre-playlist add form).
r = client.post(f"/api/bots/{bot_id}/sources/youtube", json={
    "channel": "@defaultkind", "author": "Alex", "consent": True,
})
check("add without kind defaults to channel", r.status_code == 200 and r.json().get("kind") == "channel",
      f"got {r.status_code}: {r.text[:160]}")

# a personal/auto mix in playlist mode → 400 with the mix message.
r = client.post(f"/api/bots/{bot_id}/sources/youtube", json={
    "channel": "https://www.youtube.com/playlist?list=WLpersonal",
    "author": "Alex", "consent": True, "kind": "playlist",
})
check("playlist mix (WL…) -> 400", r.status_code == 400, f"got {r.status_code}")
check("playlist mix 400 message mentions mixes",
      "mixes" in (r.json().get("error", "").lower()), r.text[:160])

# bad kind value → 400.
r = client.post(f"/api/bots/{bot_id}/sources/youtube", json={
    "channel": "@x", "author": "Alex", "consent": True, "kind": "bogus",
})
check("unknown kind -> 400", r.status_code == 400, f"got {r.status_code}")

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
