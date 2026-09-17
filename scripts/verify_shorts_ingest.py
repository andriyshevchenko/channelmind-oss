"""Offline verification for YouTube Shorts ingestion support.

Monkeypatches yt-dlp so no network is touched, then asserts:
  1. extract_channel merges /videos + /shorts DEDUPed by id.
  2. an empty (or raising) /shorts tab still yields /videos results (graceful).
  3. channel meta comes from /videos, falling back to /shorts when /videos is empty.
  4. normalize_youtube_key maps @Igor, @Igor/videos, @Igor/shorts to one key.
  5. single-video URLs pass through extract_channel unchanged (not treated as a tab).

Run: .venv/Scripts/python.exe scripts/verify_shorts_ingest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import ytrag.ingest as ingest  # noqa: E402
from ytrag.sources import normalize_youtube_key  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    RESULTS.append((name, passed, detail))


class FakeYDL:
    """Stand-in for YoutubeDL that returns canned info keyed by URL.

    ``responses`` maps a URL substring -> info dict, RuntimeError instance (to
    simulate a raising extraction), or None (empty/404 tab).
    """

    responses: dict = {}

    def __init__(self, opts):
        self.opts = opts

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download=False):
        for needle, value in self.responses.items():
            if needle in url:
                if isinstance(value, Exception):
                    raise value
                return value
        return None


def _info(title, entries):
    return {
        "channel": title,
        "thumbnails": [{"id": "avatar_uncropped", "url": f"http://img/{title}.jpg"}],
        "entries": entries,
    }


def _entry(vid, title):
    return {"id": vid, "title": title, "url": f"https://www.youtube.com/watch?v={vid}"}


def run(responses):
    FakeYDL.responses = responses
    ingest.YoutubeDL = FakeYDL  # type: ignore[assignment]
    return ingest.extract_channel("@Igor")


# --- 1 + 3: merge/dedupe, meta from /videos ---------------------------------
meta, videos = run({
    "/videos": _info("Igor", [_entry("a", "Vid A"), _entry("b", "Vid B (long)")]),
    "/shorts": _info("Igor Shorts", [_entry("b", "Vid B short-dupe"), _entry("c", "Short C")]),
})
ids = [v["id"] for v in videos]
check("merge dedupes videos+shorts to [a,b,c]", ids == ["a", "b", "c"], f"got {ids}")
b_title = next(v["title"] for v in videos if v["id"] == "b")
check("dup keeps /videos title (first/richer)", b_title == "Vid B (long)", f"got {b_title!r}")
check("channel meta from /videos", meta.get("title") == "Igor", f"got {meta.get('title')!r}")
check("avatar preserved", meta.get("avatar") == "http://img/Igor.jpg", f"got {meta.get('avatar')!r}")

# --- 2a: empty /shorts tab is graceful --------------------------------------
_meta, videos = run({
    "/videos": _info("Igor", [_entry("a", "Vid A"), _entry("b", "Vid B")]),
    "/shorts": None,  # 404 / empty tab
})
check("empty /shorts -> only /videos", [v["id"] for v in videos] == ["a", "b"],
      f"got {[v['id'] for v in videos]}")

# --- 2b: raising /shorts tab is graceful ------------------------------------
_meta, videos = run({
    "/videos": _info("Igor", [_entry("a", "Vid A")]),
    "/shorts": RuntimeError("HTTP 404"),
})
check("raising /shorts -> only /videos", [v["id"] for v in videos] == ["a"],
      f"got {[v['id'] for v in videos]}")

# --- 3b: empty /videos falls back to /shorts meta ---------------------------
meta, videos = run({
    "/videos": None,
    "/shorts": _info("Only Shorts", [_entry("c", "Short C")]),
})
check("empty /videos -> shorts results", [v["id"] for v in videos] == ["c"],
      f"got {[v['id'] for v in videos]}")
check("empty /videos -> meta falls back to /shorts", meta.get("title") == "Only Shorts",
      f"got {meta.get('title')!r}")

# --- 5: single-video URL passes through, not enumerated as a tab -------------
FakeYDL.responses = {"watch?v=solo": _info("", [_entry("solo", "Solo Video")])}
ingest.YoutubeDL = FakeYDL  # type: ignore[assignment]
_meta, videos = ingest.extract_channel("https://www.youtube.com/watch?v=solo")
check("single watch URL passthrough", [v["id"] for v in videos] == ["solo"],
      f"got {[v['id'] for v in videos]}")

FakeYDL.responses = {"/shorts/soloshort": _info("", [_entry("soloshort", "Solo Short")])}
_meta, videos = ingest.extract_channel("https://www.youtube.com/shorts/soloshort")
check("single /shorts/<id> URL passthrough", [v["id"] for v in videos] == ["soloshort"],
      f"got {[v['id'] for v in videos]}")

# --- helpers: single-video detection & base derivation ----------------------
check("_is_single_video_url(/shorts/<id>) is True", ingest._is_single_video_url("https://youtube.com/shorts/abcdef"))
check("_is_single_video_url(/shorts tab) is False", not ingest._is_single_video_url("https://youtube.com/@Igor/shorts"))
check("_channel_base strips /shorts tab", ingest._channel_base("https://www.youtube.com/@Igor/shorts") == "https://www.youtube.com/@Igor",
      ingest._channel_base("https://www.youtube.com/@Igor/shorts"))
check("_channel_base strips /videos tab", ingest._channel_base("@Igor/videos") == "https://www.youtube.com/@Igor",
      ingest._channel_base("@Igor/videos"))

# --- 4: normalize_youtube_key collapses tabs to one key ---------------------
keys = {
    normalize_youtube_key("@Igor"),
    normalize_youtube_key("@Igor/videos"),
    normalize_youtube_key("@Igor/shorts"),
    normalize_youtube_key("@Igor/streams"),
    normalize_youtube_key("https://youtube.com/@Igor/shorts"),
    normalize_youtube_key("Igor"),
}
check("normalize_youtube_key collapses all tabs to one key", keys == {"@igor"}, f"got {keys}")

# --- FIX 1: total extraction failure must RAISE, not report 0 videos ---------
def _raises(responses, channel="@Igor"):
    FakeYDL.responses = responses
    ingest.YoutubeDL = FakeYDL  # type: ignore[assignment]
    try:
        ingest.extract_channel(channel)
        return False
    except Exception:
        return True

check("both tabs raise -> extract_channel RAISES",
      _raises({"/videos": RuntimeError("proxy down"), "/shorts": RuntimeError("proxy down")}))
check("one tab raises, other has entries -> succeeds (no raise)",
      not _raises({"/videos": _info("Igor", [_entry("a", "A")]), "/shorts": RuntimeError("404")}))

# A genuinely EMPTY channel still reaches its tabs: /videos returns a real dict
# with no entries (error=None) while /shorts is absent → merges cleanly to [].
_meta, videos = run({"/videos": _info("Igor", []), "/shorts": None})
check("empty-but-reachable /videos + absent /shorts -> [] and does NOT raise",
      videos == [], f"got {videos}")

# BOTH tabs returning None is NOT a legitimately empty channel — yt-dlp only
# returns None when it couldn't fetch the page (channel-level 429/bot-block,
# network error, or both tabs 404). That must SURFACE as an error, not a false
# "done, 0 videos".
check("both tabs None (swallowed block) -> extract_channel RAISES",
      _raises({"/videos": None, "/shorts": None}))

# Single-video URL failure must also surface, not swallow into empty.
check("single-video extraction failure RAISES",
      _raises({"watch?v=boom": RuntimeError("network")}, channel="https://youtube.com/watch?v=boom"))

# --- FIX 2: non-@handle channel URL forms keep their identity ----------------
k_at = normalize_youtube_key("@Igor")
k_at_forms = {
    normalize_youtube_key("@Igor"),
    normalize_youtube_key("Igor"),
    normalize_youtube_key("https://youtube.com/@Igor/videos"),
    normalize_youtube_key("https://www.youtube.com/@Igor/shorts"),
}
check("all @handle forms fold to @igor", k_at_forms == {"@igor"}, f"got {k_at_forms}")

k_chan = {
    normalize_youtube_key("https://www.youtube.com/channel/UCabc/videos"),
    normalize_youtube_key("https://www.youtube.com/channel/UCabc/shorts"),
    normalize_youtube_key("/channel/UCabc"),
}
check("channel/UC… folds every tab to one key", k_chan == {"channel/UCabc"}, f"got {k_chan}")
check("channel/UC… key distinct from @igor", "channel/UCabc" != k_at)
check("two different /channel/UC… ids do NOT collide",
      normalize_youtube_key("/channel/UCabc") != normalize_youtube_key("/channel/UCxyz"))

k_user = normalize_youtube_key("https://www.youtube.com/user/Foo")
k_c = normalize_youtube_key("https://www.youtube.com/c/Bar")
check("/user/Foo -> user/Foo", k_user == "user/Foo", f"got {k_user!r}")
check("/c/Bar -> c/Bar", k_c == "c/Bar", f"got {k_c!r}")
check("/user/Foo stable across tab", normalize_youtube_key("/user/Foo/shorts") == "user/Foo")
distinct = {k_at, "channel/UCabc", k_user, k_c, normalize_youtube_key("/channel/UCxyz")}
check("all channel forms map to distinct keys", len(distinct) == 5, f"got {distinct}")

# --- FIX 2: _channel_base builds correct base for every form ----------------
check("_channel_base @handle", ingest._channel_base("@Igor") == "https://www.youtube.com/@Igor",
      ingest._channel_base("@Igor"))
check("_channel_base /channel/UC…",
      ingest._channel_base("https://www.youtube.com/channel/UCabc/videos")
      == "https://www.youtube.com/channel/UCabc",
      ingest._channel_base("https://www.youtube.com/channel/UCabc/videos"))
check("_channel_base /user/…",
      ingest._channel_base("https://www.youtube.com/user/Foo") == "https://www.youtube.com/user/Foo",
      ingest._channel_base("https://www.youtube.com/user/Foo"))
check("_channel_base /c/…",
      ingest._channel_base("https://www.youtube.com/c/Bar") == "https://www.youtube.com/c/Bar",
      ingest._channel_base("https://www.youtube.com/c/Bar"))
# /videos + /shorts attach onto the /channel/UC… base and extract_channel merges them.
FakeYDL.responses = {
    "channel/UCabc/videos": _info("Chan", [_entry("x", "X")]),
    "channel/UCabc/shorts": _info("Chan", [_entry("y", "Y")]),
}
ingest.YoutubeDL = FakeYDL  # type: ignore[assignment]
_meta, videos = ingest.extract_channel("https://www.youtube.com/channel/UCabc")
check("channel/UC… base: /videos+/shorts attach & merge", [v["id"] for v in videos] == ["x", "y"],
      f"got {[v['id'] for v in videos]}")

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
