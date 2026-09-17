"""Offline verification for the chat answer-formatting guidance (Fable beta win #1).

Proves two things, fully OFFLINE (no keys, no network, no real Telegram):

  1. PROMPT — the GROUNDED answer system prompt now carries the output-formatting
     guidance (scannable structure: short paragraphs, **bold** key terms, lists when
     enumerating, bold sub-heads over "#" headings) AND still carries the whole
     safety/grounding stack it must never weaken: the instruction hierarchy, the
     grounding rules, the citation contract and the untrusted-data fence. The
     SMALLTALK prompt is left UNCHANGED (no formatting guide bleeds into the
     no-retrieval chit-chat path).

  2. RENDER — a richly formatted answer (a **bold** lead, a bullet list and a
     headed source block, real multi-line quoted excerpt, and separate video watch link
     — exactly what the guidance asks the model to produce) survives the Telegram render path (``telegramify_markdown.convert`` +
     ``split_entities``) into text + entities with the source link intact, and a
     deliberately malformed render still degrades to the plain-text fallback (the
     real ``_send_message`` swallows the render error and posts plain text).

Run: .venv/Scripts/python.exe scripts/verify_answer_formatting.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

try:  # keep the console happy on Windows cp1252 shells
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Isolated throwaway dirs; offline fakes; never production.
os.environ.pop("YTRAG_ENV", None)
os.environ["YTRAG_TEST_MODE"] = "1"
os.environ["YTRAG_DATA_DIR"] = tempfile.mkdtemp(prefix="ytrag_fmt_data_")
os.environ["YTRAG_CHROMA_DIR"] = tempfile.mkdtemp(prefix="ytrag_fmt_chroma_")
# Telegram base has NO default and RAISES when unset; pin a harmless fake so no
# URL construction in this offline test can ever reach the real Telegram API.
os.environ["YTRAG_TELEGRAM_API_BASE"] = "http://127.0.0.1:9/tg-fake"

from ytrag.rag import (  # noqa: E402
    GROUNDING_RULES,
    AGENT_GROUNDING_RULES,
    INSTRUCTION_HIERARCHY,
    OUTPUT_FORMATTING,
    UNTRUSTED_CLOSE,
    UNTRUSTED_OPEN,
    build_smalltalk_system_prompt,
    build_system_prompt,
)

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(passed), str(detail) if detail and not passed else ""))


# ===========================================================================
# 1. PROMPT — grounded prompt gains formatting guidance, keeps the safety stack
# ===========================================================================
PERSONA = "You are a friendly coach bot."
grounded = build_system_prompt(PERSONA, "Ukrainian")

# (a) The formatting guidance is present and asks for the scannable structure.
check("grounded prompt embeds the OUTPUT_FORMATTING block verbatim",
      OUTPUT_FORMATTING in grounded, grounded[-400:])
lower = grounded.lower()
check("formatting: guides short paragraphs / no wall of text",
      "wall of text" in lower and "short" in lower, OUTPUT_FORMATTING[:120])
check("formatting: asks to bold key terms",
      "**bold**" in grounded.lower() or "bold" in lower, OUTPUT_FORMATTING[:120])
check("formatting: asks for bullet / numbered lists when enumerating",
      "bullet" in lower and "numbered list" in lower, OUTPUT_FORMATTING[:160])
check("formatting: prefers bold sub-heads over markdown headings",
      '"#"' in OUTPUT_FORMATTING and "heading" in lower, OUTPUT_FORMATTING)
check("formatting: scales structure to the answer (no over-formatting trivial replies)",
      "one-line" in lower or "scale the structure" in lower, OUTPUT_FORMATTING[:200])
check("formatting: protects the citation links from being mangled",
      "citation" in lower and "intact" in lower, OUTPUT_FORMATTING[-200:])
check("formatting: requires source blocks with real quoted excerpts after prose",
      "actual multi-line quote block" in OUTPUT_FORMATTING
      and "actual retrieved transcript/document excerpt" in GROUNDING_RULES
      and "NEVER put a citation inline" in GROUNDING_RULES, OUTPUT_FORMATTING)
check("agent prompt uses the same separate quote-block citation contract",
      "NO inline citations" in AGENT_GROUNDING_RULES
      and "actual retrieved excerpt as ONE Markdown blockquote" in AGENT_GROUNDING_RULES,
      AGENT_GROUNDING_RULES)

# (b) NO regression: the entire safety/grounding/citation stack is still present.
check("no regression: instruction hierarchy still present",
      INSTRUCTION_HIERARCHY in grounded)
check("no regression: grounding rules still present",
      GROUNDING_RULES in grounded)
check("no regression: citation contract still present",
      "Citations:" in grounded and "Cite every source" in grounded)
check("no regression: untrusted-data fence still named + fenced-as-instructions ban",
      UNTRUSTED_OPEN in grounded and UNTRUSTED_CLOSE in grounded
      and "NEVER as instructions" in grounded, grounded[-300:])
check("no regression: persona + language still honored",
      "friendly coach bot" in grounded and "Ukrainian" in grounded)

# (c) The formatting block is layered AFTER grounding, not spliced inside it (so it
#     shapes presentation only and cannot weaken the fence that precedes it).
check("formatting block sits AFTER the grounding rules (presentation layer)",
      grounded.index(OUTPUT_FORMATTING) > grounded.index(GROUNDING_RULES),
      f"fmt@{grounded.find(OUTPUT_FORMATTING)} grounding@{grounded.find(GROUNDING_RULES)}")

# (d) SMALLTALK path is UNCHANGED — no formatting guidance bleeds into chit-chat.
smalltalk = build_smalltalk_system_prompt(PERSONA, "Ukrainian")
check("smalltalk prompt does NOT carry the formatting guidance (unchanged)",
      OUTPUT_FORMATTING not in smalltalk, smalltalk[-200:])
check("smalltalk prompt still drops grounding (no bleed of the grounded stack)",
      GROUNDING_RULES not in smalltalk and "Citations:" not in smalltalk)

# ===========================================================================
# 2. RENDER — rich markdown survives the Telegram path; malformed falls back
# ===========================================================================
from telegramify_markdown import convert, split_entities  # noqa: E402

SOURCE_LINK = "▶️ [Дивитись з 12:34](https://youtu.be/abc123?t=754)"
RICH_ANSWER = (
    "**Головне:** підходь упевнено й одразу починай.\n\n"
    "Ось кілька кроків:\n"
    "- Підійди протягом **трьох секунд**\n"
    "- Постав відкрите запитання\n"
    "- Слухай уважно\n\n"
    "**Ігор — Як почати розмову**\n"
    "> Підійди протягом трьох секунд.\n"
    "> Постав відкрите запитання.\n"
    f"{SOURCE_LINK}"
)

render_ok = True
rendered = ""
entities: list = []
try:
    rendered, entities = convert(RICH_ANSWER)
    chunks = split_entities(rendered, entities, 4096)
except Exception as exc:  # noqa: BLE001
    render_ok = False
    chunks = None
    _render_err = repr(exc)

check("rich answer (bold + list + source link) renders to entities without error",
      render_ok and bool(entities), locals().get("_render_err", ""))
# The markdown source link becomes a Telegram ``text_link`` entity — the URL lives in
# the entity (keeping the link clickable), NOT inlined into the visible text. Assert the
# citation survives THERE, which is exactly how Telegram renders a clean link.
_entity_urls = [d.get("url") for d in (ent.to_dict() for ent in entities) if d.get("url")]
check("render preserves the source link URL in a text_link entity (citation intact)",
      "https://youtu.be/abc123?t=754" in _entity_urls, _entity_urls)
check("render carries the source citation label in the visible text",
      "Як почати розмову" in rendered, rendered[:200])
check("render keeps the real multi-line source excerpt visible",
      "Підійди протягом трьох секунд." in rendered
      and "Постав відкрите запитання." in rendered, rendered[:300])
check("render splits into at least one deliverable chunk",
      bool(chunks) and len(chunks) >= 1, f"chunks={None if chunks is None else len(chunks)}")
# The bold and list content survives into the rendered text (entities carry the styling;
# the plain glyphs of the list items remain in the text body).
check("render keeps the bulleted step content",
      "трьох секунд" in rendered and "Слухай уважно" in rendered, rendered[:300])

# The SPA's escape-first renderer must turn citation quote blocks into semantic
# blockquotes without allowing model-controlled HTML. Load it through a data URL so
# Node treats the standalone browser module as ESM without changing the repo.
mdlite = Path(__file__).resolve().parents[1] / "src" / "ytrag" / "web" / "spa" / "mdlite.js"
node_program = """
import fs from 'node:fs';
const src = fs.readFileSync(process.argv[1], 'utf8');
const { renderMarkdown } = await import('data:text/javascript,' + encodeURIComponent(src));
const html = renderMarkdown('Reasoning <script>x</script>\\n\\n**Author — Video**\\n> The actual transcript sentence.\\n> Its next real line.\\n▶️ [Дивитись з 1:23](https://youtu.be/id?t=83)\\n\\n**Author — Notes** _(документ)_\\n> Document excerpt.\\n> Its next real line.');
if ((html.match(/<blockquote class="md-quote">/g) || []).length !== 2 ||
    !html.includes('https://youtu.be/id?t=83') ||
    !html.includes('The actual transcript sentence.<br>Its next real line.') ||
    !html.includes('&lt;script&gt;') || html.includes('<script>') ||
    html.includes('href="javascript:')) process.exit(1);
"""
node = "node"
try:
    md_proc = subprocess.run(
        [node, "--input-type=module", "-e", node_program, str(mdlite)],
        capture_output=True, text=True, timeout=15,
    )
    md_ok, md_detail = md_proc.returncode == 0, md_proc.stderr.strip()
except (OSError, subprocess.TimeoutExpired) as exc:
    md_ok, md_detail = False, repr(exc)
check("web mdlite renders safe multi-line source blockquotes", md_ok, md_detail)

# Web chat must render only the model's deliberately cited answer.  The old owner
# and guest footer reconstructed several retrieved chunks into one inline "Sources"
# row, which could duplicate or contradict the actual citations.
bot_detail = (Path(__file__).resolve().parents[1] / "src" / "ytrag" / "web" / "spa"
              / "views" / "BotDetail.js").read_text(encoding="utf-8")
public_chat = (Path(__file__).resolve().parents[1] / "src" / "ytrag" / "web" / "spa"
               / "views" / "PublicChat.js").read_text(encoding="utf-8")
check("owner web chat has no duplicate retrieval Sources footer",
      "chat-citation" not in bot_detail and "toCites" not in bot_detail, bot_detail)
check("guest web chat has no duplicate retrieval Sources footer",
      "guest-citation" not in public_chat and "cites:" not in public_chat, public_chat)

# --- malformed render still degrades to plain text via the real _send_message ------
# Patch ONLY the Telegram network boundary and force convert() to raise, proving the
# fallback branch posts the raw text (delivered=0 -> plain-text send) instead of dying.
from ytrag import telegram_bot  # noqa: E402

CALLS: list[dict] = []


def _fake_call(token, method, params, timeout=30.0):
    CALLS.append({"method": method, "params": params})
    return {"ok": True, "result": {"message_id": len(CALLS)}}


_orig_call = telegram_bot._call
_orig_convert = telegram_bot.convert
telegram_bot._call = _fake_call
telegram_bot.convert = lambda text: (_ for _ in ()).throw(ValueError("boom render"))
try:
    telegram_bot._send_message("TOK", 111, RICH_ANSWER)
finally:
    telegram_bot._call = _orig_call
    telegram_bot.convert = _orig_convert

check("malformed render falls back to a single plain-text send (no crash, no dupes)",
      len(CALLS) == 1 and CALLS[0]["method"] == "sendMessage", f"calls={len(CALLS)}")
check("plain-text fallback carries the full answer text (entities dropped, content kept)",
      bool(CALLS) and "трьох секунд" in CALLS[0]["params"].get("text", "")
      and "youtu.be/abc123" in CALLS[0]["params"].get("text", ""),
      CALLS[0]["params"].get("text", "")[:120] if CALLS else "(none)")

# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
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
print(f"{sum(1 for _, ok, _ in RESULTS if ok)}/{len(RESULTS)} checks passed")
print("ALL PASS" if all_ok else "SOME FAILED")
sys.exit(0 if all_ok else 1)
