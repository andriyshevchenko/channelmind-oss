"""Bot builder: turn a plain-language description into a system prompt.

A rare, one-off call per bot, so it uses the strongest available reasoning model
(Opus 4.8) regardless of the cheaper model used for runtime chat. The generated
persona defines identity/voice/language only — the hard grounding rules that keep
answers tied to the transcripts are enforced separately at chat time (see rag.py),
so a user's description can never turn those off.
"""
from __future__ import annotations

import json

from .config import Config
from .llm import AnthropicLLM, OpenAICompatLLM

# Provider-specific slug for the builder model (strong reasoning, quality over cost).
_BUILDER_MODEL = {
    "openrouter": "anthropic/claude-opus-4.8",
    "anthropic": "claude-opus-4-8",
    "openai": "gpt-4o",  # OpenAI can't serve Claude; best in-provider fallback.
}

def _builder_llm(cfg: Config):
    from .config import test_mode

    if test_mode():
        from .testmode import FakeLLM

        return FakeLLM()
    provider = cfg.llm_provider
    model = _BUILDER_MODEL.get(provider, cfg.llm_model)
    if provider == "anthropic":
        return AnthropicLLM(cfg.anthropic_api_key, model)
    return OpenAICompatLLM(
        cfg.llm_key(), model, cfg.openai_compat_base_url(provider)
    )


_MAX_QUESTIONS = 3

# Used only when the model ignores the "ask at least one question first" rule on
# the opening turn and gives us no question of its own — keeps the live interview
# from silently collapsing to a one-shot persona.
_FIRST_TURN_FALLBACK_QUESTION = (
    "To tailor your bot's voice: what tone and personality should it have "
    "(for example formal and factual, warm and encouraging, or witty and "
    "playful), and which language should it reply in?"
)

_INTERVIEW_PROMPT = (
    "You are an expert prompt engineer designing a SYSTEM PROMPT for a chatbot that "
    "answers questions grounded in the transcripts of one YouTube channel.\n\n"
    "You may ask the user UP TO 3 short clarifying questions, ONE at a time, to nail "
    "down the bot's identity, tone, language, and scope. Ask only what you genuinely "
    "need.\n\n"
    "ALWAYS ask at least one clarifying question before finishing: on the very "
    "first turn (when no questions have been answered yet) you MUST return "
    "done=false with a single tailored question — never finish immediately. From "
    "the second turn onward you may finish early if the description plus answers "
    "already suffice.\n\n"
    "You will be given the original description and any prior questions/answers. "
    "Respond with STRICT JSON ONLY — no prose, no markdown fences — matching:\n"
    '{"done": bool, "question": string|null, "persona": string|null}\n\n'
    "- If you need more info AND fewer than 3 questions have been asked so far: "
    'set done=false, question=<your next single clarifying question>, persona=null.\n'
    "- Otherwise: set done=true, question=null, persona=<the final system prompt>.\n\n"
    "The final persona must follow these rules:\n"
    "- Written in second person ('You are...'), no preamble, no markdown headers, no "
    "explanations.\n"
    "- Capture identity (who the bot is / which channel it represents), personality & "
    "tone, the language it must reply in, and scope & behavior (including graceful "
    "handling of greetings/off-topic messages).\n"
    "- Do NOT include retrieval/citation/anti-hallucination instructions; the platform "
    "adds those automatically.\n"
    "- Keep it concise and directive (roughly 120-250 words)."
)


def _parse_interview_reply(text: str) -> dict:
    """Parse the model's strict-JSON reply, tolerating stray ```json fences.

    On any failure, fall back to treating the whole reply as a finished persona so
    the UI never dead-ends.
    """
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        # Drop opening fence (optionally ```json) and closing fence.
        cleaned = cleaned[3:]
        if cleaned[:4].lower() == "json":
            cleaned = cleaned[4:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
    try:
        data = json.loads(cleaned)
        if not isinstance(data, dict):
            raise ValueError("not an object")
        return {
            "done": bool(data.get("done")),
            "question": data.get("question"),
            "persona": data.get("persona"),
        }
    except Exception:  # noqa: BLE001
        return {"done": True, "question": None, "persona": (text or "").strip()}


def interview_persona(cfg: Config, description: str, answers: list[dict]) -> dict:
    """Drive a multi-turn interview that yields a system prompt.

    ``answers`` is the prior Q&A as a list of {"question", "answer"} dicts (empty on
    the first call). Returns {"done", "question", "persona"}.
    """
    description = (description or "").strip()
    if not description:
        raise ValueError("description is required")
    answers = answers or []

    lines = [f"Original description:\n{description}\n"]
    if answers:
        lines.append("Prior questions and answers:")
        for i, qa in enumerate(answers, 1):
            q = (qa.get("question") or "").strip()
            a = (qa.get("answer") or "").strip()
            lines.append(f"{i}. Q: {q}\n   A: {a}")
        lines.append("")

    if len(answers) >= _MAX_QUESTIONS:
        lines.append(
            "You have now asked the maximum of 3 questions. You MUST finish now: "
            'return done=true, question=null, and a complete persona. Do NOT ask '
            "another question."
        )
    elif not answers:
        lines.append(
            "This is the FIRST turn and the user has not answered anything yet. You "
            "MUST ask exactly one clarifying question now: return done=false, "
            "question=<one tailored clarifying question>, persona=null. Do NOT "
            "finish on this turn."
        )
    else:
        remaining = _MAX_QUESTIONS - len(answers)
        lines.append(
            f"You may ask at most {remaining} more question(s). Ask one now only if "
            "you truly need it; otherwise finish with the persona."
        )

    llm = _builder_llm(cfg)
    messages = [{"role": "user", "content": "\n".join(lines)}]
    reply = llm.complete(_INTERVIEW_PROMPT, messages, max_tokens=1200).text
    result = _parse_interview_reply(reply)

    # Guarantee the live interview actually happens: on the FIRST turn we must ask
    # at least one clarifying question, even if the model ignored the prompt and
    # tried to finish immediately (common for detailed descriptions). Prefer any
    # question the model did volunteer; otherwise fall back to a tailored default.
    if not answers:
        question = (result.get("question") or "").strip()
        if result["done"] or not question:
            result = {
                "done": False,
                "question": question or _FIRST_TURN_FALLBACK_QUESTION,
                "persona": None,
            }

    # Enforce the hard cap even if the model ignores the instruction.
    if len(answers) >= _MAX_QUESTIONS and not result["done"]:
        result = {
            "done": True,
            "question": None,
            "persona": result.get("persona") or reply.strip(),
        }
    return result
