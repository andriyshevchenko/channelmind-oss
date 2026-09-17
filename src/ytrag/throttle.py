"""Adaptive, patient throttling so bulk ingestion never trips YouTube's rate limits.

Strategy: pace every request with a randomized base delay. On any block/error,
multiply the delay (exponential backoff, capped) and sleep a long cool-off; on
sustained success, slowly relax back toward the base delay. This trades speed
for never getting rate-limited — a channel can take hours or days but keeps going.
"""
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ThrottleConfig:
    base_delay: float = 2.0        # seconds between successful requests (before jitter)
    jitter: float = 0.5           # +/- fraction of base_delay added randomly
    max_delay: float = 900.0      # cap on the paced delay (15 min)
    backoff_factor: float = 2.0   # multiply current delay on each failure
    cooloff_min: float = 30.0     # extra sleep floor after a block
    cooloff_max: float = 1800.0   # extra sleep ceiling after a block (30 min)
    recover_factor: float = 0.9   # multiply delay by this after each success (decay)


class AdaptiveThrottle:
    def __init__(self, cfg: ThrottleConfig | None = None):
        self.cfg = cfg or ThrottleConfig()
        self._delay = self.cfg.base_delay
        self.consecutive_failures = 0

    def wait(self, operation: str = "request") -> None:
        """Sleep the current paced delay (with jitter) before the next request."""
        j = self._delay * self.cfg.jitter
        delay = max(0.0, self._delay + random.uniform(-j, j))
        # DEBUG only: this runs once per video — never surface pacing at INFO.
        logger.debug("throttle[%s]: pacing %.1fs before next request", operation, delay)
        time.sleep(delay)

    def on_success(self, operation: str = "request") -> None:
        # If we were backing off, note that requests are flowing again (INFO —
        # a state transition, and the "are we still blocked?" answer operators want).
        if self.consecutive_failures:
            logger.info(
                "throttle[%s]: recovered after %d consecutive block(s); resuming (delay=%.1fs)",
                operation, self.consecutive_failures, self._delay,
            )
        self.consecutive_failures = 0
        self._delay = max(self.cfg.base_delay, self._delay * self.cfg.recover_factor)

    def on_block(self, operation: str = "request") -> float:
        """Escalate delay and return a long cool-off sleep (seconds) to observe."""
        self.consecutive_failures += 1
        self._delay = min(self.cfg.max_delay, self._delay * self.cfg.backoff_factor)
        cooloff = min(
            self.cfg.cooloff_max,
            self.cfg.cooloff_min * (self.cfg.backoff_factor ** (self.consecutive_failures - 1)),
        )
        total = cooloff + random.uniform(0, self.cfg.cooloff_min)
        # WARNING: a 429/bot-check/block is exactly what an operator staring at a
        # stuck import needs to see — with the backoff now applied.
        logger.warning(
            "throttle[%s]: block/rate-limit detected (block #%d) — backing off: "
            "delay=%.1fs, cooling off ~%s",
            operation, self.consecutive_failures, self._delay, humanize_seconds(total),
        )
        return total

    @property
    def current_delay(self) -> float:
        return self._delay


_BLOCK_MARKERS = (
    "http error 429",
    "too many requests",
    "sign in to confirm you're not a bot",
    "requestblocked",
    "ipblocked",
    "rate limit",
    "temporarily unavailable",
)

# Permanent per-video failures: retrying never helps, so skip immediately even in
# unlimited-patience mode (otherwise the run would loop forever on one video).
_PERMANENT_MARKERS = (
    "sign in to confirm your age",
    "age-restricted",
    "inappropriate for some users",
    "private video",
    "members-only",
    "this video is available to this channel's members",
    "video unavailable",
    "removed by the uploader",
    "account associated with this video has been terminated",
)


def is_permanent_skip(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(m in msg for m in _PERMANENT_MARKERS)


def is_block_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    if is_permanent_skip(exc):
        return False
    if any(m in msg for m in _BLOCK_MARKERS):
        return True
    # Generic "blocked" only if it isn't a permanent condition handled above.
    return "blocked" in msg


# Transient TRANSPORT failures: a dropped socket / TLS EOF / timeout from a flaky
# residential proxy exit (or a momentary network blip), NOT a YouTube block and NOT
# a permanent per-video condition. Retrying helps — but only if the retry leaves via
# a DIFFERENT exit IP (see the caller's sticky-session rotation), otherwise it keeps
# hitting the same broken exit. Kept deliberately narrow: only markers that mean "the
# pipe broke", never anything that could be a real 403/permanent condition.
_TRANSPORT_MARKERS = (
    "unexpected_eof",
    "eof occurred in violation of protocol",
    "ssl:",
    "connection reset",
    "connection aborted",
    "connection refused",
    "remote end closed connection",
    "timed out",
    "temporary failure in name resolution",
    "unable to connect to proxy",
    "tunnel connection failed",
)


def is_transient_transport_error(exc: Exception) -> bool:
    """A network/TLS hiccup (dead residential exit, dropped socket) — retryable
    with a FRESH proxy session, distinct from a YouTube block or a permanent
    per-video condition. Block/permanent take precedence so a 429 tunnelled inside
    an SSL message is still treated as a block."""
    if is_permanent_skip(exc) or is_block_error(exc):
        return False
    msg = str(exc).lower()
    return any(m in msg for m in _TRANSPORT_MARKERS)


def humanize_seconds(seconds: float) -> str:
    """Format a duration as a compact human string, e.g. '2m 5s', '1h 3m', '2d 4h'."""
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    d, rem = divmod(s, 86400)
    h, rem = divmod(rem, 3600)
    m, sec = divmod(rem, 60)
    parts = []
    if d:
        parts += [f"{d}d", f"{h}h"]
    elif h:
        parts += [f"{h}h", f"{m}m"]
    else:
        parts += [f"{m}m", f"{sec}s"]
    return " ".join(parts)
