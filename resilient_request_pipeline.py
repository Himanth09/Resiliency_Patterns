"""
resilient_request_pipeline.py

Implements the request-handling flow:

    Client Request
        -> Authentication
        -> Authorization
        -> Idempotency Check
        -> Rate Limit
        -> Dependency Call
             -> Success -> Final Result
             -> Failure -> Is it retryable?
                    -> YES -> Backoff -> Retry -> Circuit Breaker -> (loop back to Dependency Call
                                                                       or eventually Fail)
                    -> NO  -> Fail

Design goals (the things an interviewer / reviewer will look for):
    1. Each pipeline stage is a separate, single-responsibility component so it can be
       unit tested and swapped out independently (e.g., swap InMemoryRateLimiter for a
       Redis-backed one without touching the rest of the pipeline).
    2. Failures are modeled as typed exceptions so the orchestrator can make routing
       decisions (retryable vs non-retryable) without string-matching error messages.
    3. Retry uses exponential backoff WITH JITTER to avoid the "thundering herd" problem
       where many clients retry at exactly the same intervals after a shared dependency
       recovers.
    4. A circuit breaker sits around the dependency call so that once a dependency is
       clearly unhealthy, we stop hammering it with retries (which would make recovery
       slower) and fail fast instead, giving it time to recover.
    5. Idempotency check prevents duplicate side effects if a client retries a request
       that actually succeeded server-side but the response was lost in transit.
    6. Everything is logged at each stage transition, which is what you'd wire into
       CloudWatch / Splunk / Datadog in a real production system for observability.
"""

from __future__ import annotations

import concurrent.futures
import logging
import random
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional


# --------------------------------------------------------------------------------------
# Logging setup
# --------------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("resilient_pipeline")


# ========================================================================================
# 1. DOMAIN EXCEPTIONS
# ----------------------------------------------------------------------------------------
# Typed exceptions let the orchestrator branch on failure TYPE rather than inspecting
# error strings. This is the same pattern you'd use to decide "retryable vs not" in a
# real SRE/production-support system.
# ========================================================================================

class PipelineError(Exception):
    """Base class for all errors raised within the pipeline."""


class AuthenticationError(PipelineError):
    """Raised when the caller's identity cannot be verified (e.g., bad/expired token)."""


class AuthorizationError(PipelineError):
    """Raised when the caller IS who they say they are, but isn't allowed to do this."""


class RateLimitExceededError(PipelineError):
    """
    Raised when the caller has exceeded their allotted request rate.
    `retry_after_sec` lets the HTTP layer set a precise `Retry-After` response header
    instead of the client guessing/polling.
    """

    def __init__(self, message: str, retry_after_sec: float = 0.0):
        super().__init__(message)
        self.retry_after_sec = retry_after_sec


class IdempotencyConflictError(PipelineError):
    """
    Raised when a request with the same idempotency key is already IN-FLIGHT
    (as opposed to already completed, which is handled by returning the cached result).
    """


class DependencyError(PipelineError):
    """
    Raised when the downstream dependency call fails.
    `retryable` tells the orchestrator whether this class of failure is safe to retry
    (e.g., timeout, 503) vs not (e.g., 400 Bad Request -- retrying won't help).
    """

    def __init__(self, message: str, retryable: bool, status_code: Optional[int] = None):
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code


class CircuitOpenError(PipelineError):
    """Raised when the circuit breaker is OPEN and is short-circuiting calls (fail fast)."""


# ========================================================================================
# 2. REQUEST CONTEXT
# ----------------------------------------------------------------------------------------
# A simple envelope carrying everything that flows through the pipeline stages.
# In a real system this might be a pydantic model / dataclass parsed from an HTTP request.
# ========================================================================================

@dataclass
class RequestContext:
    request_id: str
    caller_token: str
    caller_id: Optional[str] = None          # populated by Authentication
    scopes: Optional[list] = None             # populated by Authentication
    idempotency_key: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    attempt: int = 0                          # incremented on each retry


# ========================================================================================
# 3. AUTHENTICATION
# ----------------------------------------------------------------------------------------
# Verifies WHO is calling. In production this would validate a JWT signature/expiry
# against an identity provider (e.g., Okta, Cognito, internal auth service) -- here it's
# stubbed with a simple token->identity lookup to keep the example self-contained.
# ========================================================================================

class Authenticator:
    def __init__(self, valid_tokens: Dict[str, str]):
        """
        :param valid_tokens: mapping of token -> caller_id (stand-in for a real
                              token-introspection / JWT-verification call).
        """
        self._valid_tokens = valid_tokens

    def authenticate(self, ctx: RequestContext) -> None:
        logger.info("[%s] Authenticating caller...", ctx.request_id)

        caller_id = self._valid_tokens.get(ctx.caller_token)
        if caller_id is None:
            logger.warning("[%s] Authentication FAILED (invalid/expired token)", ctx.request_id)
            raise AuthenticationError("Invalid or expired token")

        ctx.caller_id = caller_id
        logger.info("[%s] Authentication OK -> caller_id=%s", ctx.request_id, caller_id)


# ========================================================================================
# 4. AUTHORIZATION
# ----------------------------------------------------------------------------------------
# Verifies WHAT the (now-known) caller is allowed to do. Kept separate from
# Authentication deliberately -- these are different concerns with different failure
# semantics (401 vs 403 in HTTP terms) and often different systems of record
# (identity provider vs internal entitlements/RBAC service).
# ========================================================================================

class Authorizer:
    def __init__(self, entitlements: Dict[str, list]):
        """
        :param entitlements: mapping of caller_id -> list of granted scopes/permissions.
        """
        self._entitlements = entitlements

    def authorize(self, ctx: RequestContext, required_scope: str) -> None:
        logger.info(
            "[%s] Authorizing caller_id=%s for scope=%s...",
            ctx.request_id, ctx.caller_id, required_scope,
        )

        granted_scopes = self._entitlements.get(ctx.caller_id, [])
        ctx.scopes = granted_scopes

        if required_scope not in granted_scopes:
            logger.warning(
                "[%s] Authorization FAILED -- caller_id=%s lacks scope=%s",
                ctx.request_id, ctx.caller_id, required_scope,
            )
            raise AuthorizationError(
                f"Caller '{ctx.caller_id}' lacks required scope '{required_scope}'"
            )

        logger.info("[%s] Authorization OK", ctx.request_id)


# ========================================================================================
# 5. IDEMPOTENCY CHECK
# ----------------------------------------------------------------------------------------
# Prevents duplicate side effects when a client retries a request (e.g., due to a
# network timeout on their end even though the server actually completed it).
#
# Pattern:
#   - Client sends an Idempotency-Key header with every logically-identical request.
#   - Server checks a store (in production: Redis/DynamoDB with a TTL) for that key:
#       * Not found            -> proceed, and mark key as IN_PROGRESS
#       * IN_PROGRESS          -> another instance of this exact request is currently
#                                  being processed concurrently -> reject/conflict
#       * COMPLETED (cached)   -> return the cached result immediately, do NOT re-run
#                                  the dependency call (this is what prevents duplicate
#                                  side effects like double-charging a payment)
#
# A real implementation needs this store to be distributed (shared across all instances
# of the service) and to use an atomic "set if not exists" operation to avoid a race
# where two concurrent requests both see "not found" and both proceed.
# ========================================================================================

class IdempotencyStatus(Enum):
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"


class IdempotencyStore:
    """
    Thread-safe in-memory idempotency store.
    Swap this out for a Redis/DynamoDB-backed implementation in production --
    the interface (check_and_lock / mark_completed / get_cached_result) stays the same.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._store: Dict[str, Dict[str, Any]] = {}
        # Structure: { idempotency_key: {"status": IdempotencyStatus, "result": Any} }

    def check_and_lock(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Atomically checks the current state for `key` and, if not already tracked,
        marks it IN_PROGRESS. Returns the existing entry if one is found (so the
        caller can decide whether to short-circuit), or None if this is a fresh key
        that was just locked for this caller to proceed with.
        """
        with self._lock:
            existing = self._store.get(key)
            if existing is not None:
                return existing

            self._store[key] = {"status": IdempotencyStatus.IN_PROGRESS, "result": None}
            return None

    def mark_completed(self, key: str, result: Any) -> None:
        with self._lock:
            self._store[key] = {"status": IdempotencyStatus.COMPLETED, "result": result}

    def release_on_failure(self, key: str) -> None:
        """
        If the request ultimately fails (after retries are exhausted), release the lock
        so a future retry with the SAME idempotency key can attempt again, rather than
        being permanently stuck as IN_PROGRESS.
        """
        with self._lock:
            self._store.pop(key, None)


class IdempotencyChecker:
    def __init__(self, store: IdempotencyStore):
        self._store = store

    def check(self, ctx: RequestContext) -> Optional[Any]:
        """
        Returns a cached result if this exact request already completed successfully.
        Returns None if this is a new request that should proceed through the pipeline.
        Raises IdempotencyConflictError if an identical request is currently in-flight.
        """
        if ctx.idempotency_key is None:
            # No idempotency key provided -- caller has opted out of this protection.
            logger.info("[%s] No idempotency key provided, skipping check", ctx.request_id)
            return None

        logger.info(
            "[%s] Checking idempotency for key=%s...", ctx.request_id, ctx.idempotency_key
        )

        existing = self._store.check_and_lock(ctx.idempotency_key)

        if existing is None:
            logger.info("[%s] Idempotency key is new -- proceeding", ctx.request_id)
            return None

        if existing["status"] == IdempotencyStatus.COMPLETED:
            logger.info(
                "[%s] Idempotency key already COMPLETED -- returning cached result",
                ctx.request_id,
            )
            return existing["result"]

        # status == IN_PROGRESS
        logger.warning(
            "[%s] Idempotency key currently IN_PROGRESS elsewhere -- rejecting",
            ctx.request_id,
        )
        raise IdempotencyConflictError(
            f"A request with idempotency key '{ctx.idempotency_key}' is already in progress"
        )


# ========================================================================================
# 6. RATE LIMITER
# ----------------------------------------------------------------------------------------
# Token bucket algorithm: each caller has a bucket that refills at a fixed rate and has
# a maximum capacity (burst allowance). Each request consumes one token; if the bucket
# is empty, the request is rejected.
#
# Token bucket is preferred over a naive "fixed window counter" because it allows
# controlled bursts while still enforcing a smooth average rate, and it doesn't have
# the "boundary burst" problem of fixed windows (e.g., 2x the limit sneaking through
# right at a window boundary).
# ========================================================================================

class TokenBucketRateLimiter:
    """
    IMPROVEMENT vs. original version: per-caller locking instead of one global lock.

    ACTION: the original implementation guarded the ENTIRE bucket dict with a single
    `self._lock`, so caller A checking their rate limit blocked caller B from checking
    theirs at the same instant, even though they don't share any state. Under load with
    many distinct callers, this single lock becomes a throughput bottleneck and serializes
    otherwise-independent requests.

    FIX: keep one lock PER CALLER (created on first use, guarded briefly by a small
    "creation lock" only for the instant a new caller's bucket is first allocated).
    After that, each caller's own lock protects only their own bucket, so concurrent
    requests from different callers no longer contend with each other at all.

    Also added: `time_until_next_token()` so the caller-facing layer can return a
    `Retry-After` value to the client instead of a bare rejection -- this lets well-behaved
    clients back off precisely rather than immediately hammering the endpoint again.
    """

    def __init__(self, capacity: int, refill_rate_per_sec: float):
        """
        :param capacity: max tokens (i.e., max burst size) per caller.
        :param refill_rate_per_sec: tokens added back per second (i.e., sustained rate).
        """
        self._capacity = capacity
        self._refill_rate = refill_rate_per_sec
        self._buckets: Dict[str, Dict[str, Any]] = {}
        # Guards creation of a new per-caller entry only -- held very briefly.
        self._creation_lock = threading.Lock()

    def _get_bucket(self, caller_id: str) -> Dict[str, Any]:
        bucket = self._buckets.get(caller_id)
        if bucket is not None:
            return bucket

        with self._creation_lock:
            # Re-check inside the lock in case another thread created it first
            # (classic double-checked locking to avoid a race on first access).
            bucket = self._buckets.get(caller_id)
            if bucket is None:
                bucket = {
                    "tokens": float(self._capacity),
                    "last_refill": time.monotonic(),
                    "lock": threading.Lock(),
                }
                self._buckets[caller_id] = bucket
            return bucket

    def _refill(self, bucket: Dict[str, Any]) -> None:
        now = time.monotonic()
        elapsed = now - bucket["last_refill"]
        refill_amount = elapsed * self._refill_rate
        bucket["tokens"] = min(self._capacity, bucket["tokens"] + refill_amount)
        bucket["last_refill"] = now

    def allow(self, caller_id: str) -> bool:
        bucket = self._get_bucket(caller_id)
        with bucket["lock"]:  # only blocks THIS caller's own concurrent requests
            self._refill(bucket)

            if bucket["tokens"] >= 1.0:
                bucket["tokens"] -= 1.0
                return True
            return False

    def time_until_next_token(self, caller_id: str) -> float:
        """
        Returns the number of seconds until at least one token will be available for
        this caller. Used to populate a `Retry-After` header so throttled clients know
        precisely how long to wait instead of guessing / retrying immediately.
        """
        bucket = self._get_bucket(caller_id)
        with bucket["lock"]:
            self._refill(bucket)
            if bucket["tokens"] >= 1.0:
                return 0.0
            tokens_needed = 1.0 - bucket["tokens"]
            return tokens_needed / self._refill_rate


class RateLimiterGuard:
    def __init__(self, limiter: TokenBucketRateLimiter):
        self._limiter = limiter

    def check(self, ctx: RequestContext) -> None:
        logger.info("[%s] Checking rate limit for caller_id=%s...", ctx.request_id, ctx.caller_id)

        if not self._limiter.allow(ctx.caller_id):
            # IMPROVEMENT: attach a precise retry_after (seconds) to the exception instead
            # of a bare rejection, so the HTTP layer can return a `Retry-After` header and
            # well-behaved clients know exactly when to try again rather than polling blindly.
            retry_after = self._limiter.time_until_next_token(ctx.caller_id)
            logger.warning(
                "[%s] Rate limit EXCEEDED for caller_id=%s (retry_after=%.2fs)",
                ctx.request_id, ctx.caller_id, retry_after,
            )
            raise RateLimitExceededError(
                f"Rate limit exceeded for caller '{ctx.caller_id}'. Retry after {retry_after:.2f}s",
                retry_after_sec=retry_after,
            )

        logger.info("[%s] Rate limit check OK", ctx.request_id)


# ========================================================================================
# 7. CIRCUIT BREAKER
# ----------------------------------------------------------------------------------------
# Classic three-state circuit breaker (CLOSED / OPEN / HALF_OPEN):
#
#   CLOSED     - normal operation, calls pass through. Failures are counted in a
#                rolling window; if failure count crosses `failure_threshold`, trip to OPEN.
#   OPEN       - calls are short-circuited immediately (fail fast) without even attempting
#                the dependency call, for `reset_timeout_sec`. This protects the already
#                struggling dependency from more load and gives it room to recover.
#   HALF_OPEN  - after the reset timeout, allow a small number of "trial" calls through.
#                If they succeed, close the circuit (resume normal operation).
#                If they fail, re-open the circuit and wait again.
#
# This is what prevents the "retry storm" failure mode: without a circuit breaker, every
# client aggressively retrying a struggling dependency can make an outage worse and delay
# recovery. The circuit breaker converts "keep hammering it" into "back off and let it heal."
# ========================================================================================

class CircuitState(Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreaker:
    """
    IMPROVEMENT 1 -- rolling time-window failure counting instead of consecutive-failure
    counting.

    ACTION: the original version tracked a simple `_failure_count` that incremented on
    failure and reset to 0 on ANY success. That means a dependency that's genuinely
    unhealthy but intermittently flaky -- e.g., fail, succeed, fail, succeed, at a ~50%
    error rate -- would NEVER trip the breaker, because every other success zeroed the
    counter before it could reach `failure_threshold`. This is a real gap: intermittent
    failure is one of the most common real-world dependency degradation patterns (e.g.,
    one bad instance behind a load balancer being hit ~50% of the time).

    FIX: track failure TIMESTAMPS in a rolling window (`failure_window_sec`) using a
    deque, pruning entries older than the window on every check. The breaker now trips
    when the count of failures WITHIN the window crosses the threshold, regardless of
    whether successes were interleaved -- correctly catching a sustained elevated error
    rate, not just an unbroken streak of failures.

    IMPROVEMENT 2 -- require multiple consecutive successes in HALF_OPEN before fully
    closing, instead of just one.

    ACTION: previously a single lucky trial call succeeding would immediately flip the
    breaker back to CLOSED and resume full traffic. One success is weak evidence of real
    recovery -- it could just as easily be a lucky retry against a still-degraded backend.

    FIX: added `success_threshold_half_open` (default 2) -- the breaker now needs that
    many consecutive trial successes in HALF_OPEN before fully closing. Any failure during
    the trial period immediately re-opens it.

    IMPROVEMENT 3 -- jitter on the reset timeout.

    ACTION: with a fixed `reset_timeout_sec`, every instance of this service (if this
    breaker's state were shared, e.g., via Redis, across a fleet) would transition to
    HALF_OPEN at exactly the same moment and send trial calls simultaneously -- a smaller
    version of the same thundering-herd problem retries have without jitter.

    FIX: reset timeout is now `reset_timeout_sec +/- jitter`, randomized per trip, so
    trial calls from different breaker instances spread out instead of syncing up.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        failure_window_sec: float = 30.0,
        reset_timeout_sec: float = 30.0,
        reset_timeout_jitter_sec: float = 5.0,
        half_open_trial_calls: int = 1,
        success_threshold_half_open: int = 2,
    ):
        self._failure_threshold = failure_threshold
        self._failure_window_sec = failure_window_sec
        self._reset_timeout_sec = reset_timeout_sec
        self._reset_timeout_jitter_sec = reset_timeout_jitter_sec
        self._half_open_trial_calls = half_open_trial_calls
        self._success_threshold_half_open = success_threshold_half_open

        self._state = CircuitState.CLOSED
        self._failure_timestamps: deque = deque()   # rolling window of failure times
        self._half_open_successes = 0
        self._opened_at: Optional[float] = None
        self._current_reset_timeout: float = reset_timeout_sec
        self._half_open_calls_in_flight = 0
        self._lock = threading.Lock()

    def _prune_old_failures(self) -> None:
        cutoff = time.monotonic() - self._failure_window_sec
        while self._failure_timestamps and self._failure_timestamps[0] < cutoff:
            self._failure_timestamps.popleft()

    def _maybe_transition_to_half_open(self) -> None:
        """If we've been OPEN long enough (with jitter), allow trial calls through."""
        if self._state == CircuitState.OPEN and self._opened_at is not None:
            if (time.monotonic() - self._opened_at) >= self._current_reset_timeout:
                logger.info("Circuit breaker transitioning OPEN -> HALF_OPEN (trial period)")
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls_in_flight = 0
                self._half_open_successes = 0

    def allow_request(self) -> bool:
        with self._lock:
            self._maybe_transition_to_half_open()

            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                return False  # fail fast -- do not even attempt the dependency call

            if self._state == CircuitState.HALF_OPEN:
                # Only allow a limited number of concurrent trial calls through.
                if self._half_open_calls_in_flight < self._half_open_trial_calls:
                    self._half_open_calls_in_flight += 1
                    return True
                return False

            return False  # unreachable, keeps type checkers happy

    def record_success(self) -> None:
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_successes += 1
                logger.info(
                    "Circuit breaker trial call succeeded (%d/%d needed to close)",
                    self._half_open_successes, self._success_threshold_half_open,
                )
                if self._half_open_successes >= self._success_threshold_half_open:
                    logger.info("Circuit breaker HALF_OPEN -> CLOSED (recovery confirmed)")
                    self._state = CircuitState.CLOSED
                    self._failure_timestamps.clear()
                    self._opened_at = None
                    self._half_open_calls_in_flight = 0
                    self._half_open_successes = 0
                else:
                    # Allow another trial call through; stay in HALF_OPEN.
                    self._half_open_calls_in_flight = 0
            elif self._state == CircuitState.CLOSED:
                # A success doesn't wipe the rolling window (that would recreate the
                # "consecutive-only" blind spot) -- old failures simply age out on their own.
                pass

    def record_failure(self) -> None:
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                logger.warning("Circuit breaker trial call FAILED -> HALF_OPEN -> OPEN (retry timeout)")
                self._trip_open()
                return

            self._prune_old_failures()
            self._failure_timestamps.append(time.monotonic())

            if len(self._failure_timestamps) >= self._failure_threshold:
                logger.warning(
                    "Circuit breaker failure threshold reached (%d failures within %.0fs window) "
                    "-> CLOSED -> OPEN",
                    len(self._failure_timestamps), self._failure_window_sec,
                )
                self._trip_open()

    def _trip_open(self) -> None:
        self._state = CircuitState.OPEN
        self._opened_at = time.monotonic()
        self._half_open_calls_in_flight = 0
        self._half_open_successes = 0
        # Jittered reset timeout -- see IMPROVEMENT 3 in the class docstring.
        self._current_reset_timeout = self._reset_timeout_sec + random.uniform(
            0, self._reset_timeout_jitter_sec
        )

    @property
    def state(self) -> CircuitState:
        with self._lock:
            self._maybe_transition_to_half_open()
            return self._state


# ========================================================================================
# 8. RETRY WITH EXPONENTIAL BACKOFF + JITTER
# ----------------------------------------------------------------------------------------
# Exponential backoff: wait time doubles (roughly) with each attempt, capped at a max.
# Jitter: randomize the wait time within a range, rather than using the exact computed
# value. This is critical at scale -- without jitter, many clients that all failed at
# the same time (e.g., due to a shared dependency blip) will all retry at exactly the
# same synchronized intervals, creating repeated load spikes ("thundering herd") on the
# recovering dependency. "Full jitter" (randomizing uniformly between 0 and the computed
# max delay) is the AWS-recommended approach and is what's used here.
# ========================================================================================

def compute_backoff_delay(
    attempt: int,
    base_delay_sec: float = 0.5,
    max_delay_sec: float = 10.0,
) -> float:
    """
    Full-jitter exponential backoff, per AWS Architecture Blog's guidance:
    https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/

    delay = random_between(0, min(max_delay, base_delay * 2^attempt))
    """
    exponential_delay = min(max_delay_sec, base_delay_sec * (2 ** attempt))
    jittered_delay = random.uniform(0, exponential_delay)
    return jittered_delay


# ========================================================================================
# 9. DEPENDENCY CALL (stubbed downstream service)
# ----------------------------------------------------------------------------------------
# Stands in for a real downstream call (e.g., an HTTP call to another microservice, a
# database write, a third-party API call). Includes a simple simulated failure mode so
# the retry/circuit-breaker logic has something to exercise when you run this file.
# ========================================================================================

class DependencyClient:
    """
    IMPROVEMENT -- explicit, enforced timeout around the dependency call.

    ACTION (the gap): the original `call()` had NO timeout at all. It relied entirely on
    whatever the underlying HTTP client's default behavior happened to be. In practice
    this is a common, easy-to-miss production bug: a library's default timeout is often
    `None` (wait forever) unless you explicitly set one -- `requests.post(...)` with no
    `timeout=` argument will hang indefinitely on a connection that accepts the TCP
    handshake but never responds. Worse, this silently defeats the entire resilience
    pattern built around it: a hung call never raises an exception, so it never reaches
    the retry logic, never gets counted as a circuit-breaker failure, and the caller's
    request thread is blocked forever. All the retry/backoff/circuit-breaker machinery
    upstream is useless if the call it wraps can simply never return.

    FIX: two layers of timeout enforcement, matching real production practice:
      1. CONNECT timeout vs READ timeout modeled separately (`connect_timeout_sec`,
         `read_timeout_sec`) -- these are genuinely different failure modes. A slow/dead
         network path (connect) usually means "try a different host/retry sooner." A slow
         *application* response (read) after a successful connection often means the
         downstream service is overloaded -- both are retryable, but distinguishing them
         is useful for diagnostics (which is surfaced in the log message and the
         DependencyError's `status_code`-equivalent categorization).
      2. A hard enforced wall-clock timeout via `concurrent.futures`, independent of
         whatever the underlying client library does. This is defense-in-depth: even if
         someone later swaps in an HTTP client or SDK call that doesn't respect its own
         timeout parameter correctly (this happens more often than it should, e.g. with
         some SDKs' retry-wrapped clients silently ignoring a passed timeout), the outer
         enforced timeout still guarantees the call cannot hang the pipeline forever.

    A timeout is always classified as `retryable=True` -- a slow response now doesn't
    mean the dependency won't succeed on a fresh attempt (especially behind a load
    balancer with multiple backend instances).
    """

    def __init__(
        self,
        simulated_failure_rate: float = 0.0,
        simulated_hang_rate: float = 0.0,
        connect_timeout_sec: float = 2.0,
        read_timeout_sec: float = 5.0,
    ):
        """
        :param simulated_failure_rate: 0.0-1.0 probability the call fails fast, for demo purposes.
        :param simulated_hang_rate: 0.0-1.0 probability the call hangs indefinitely, for demo
                                     purposes -- exercises the enforced-timeout path specifically.
        :param connect_timeout_sec: max time to wait for the connection to be established.
        :param read_timeout_sec: max time to wait for a response after connecting.
        """
        self._simulated_failure_rate = simulated_failure_rate
        self._simulated_hang_rate = simulated_hang_rate
        self._connect_timeout_sec = connect_timeout_sec
        self._read_timeout_sec = read_timeout_sec
        # Bounded pool -- prevents unbounded thread creation if many calls happen
        # concurrently; excess calls simply queue for a worker rather than spawning
        # unlimited OS threads.
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=20, thread_name_prefix="dependency-call"
        )

    def _do_call(self, ctx: RequestContext) -> Dict[str, Any]:
        """
        The actual (simulated) network call. Real implementation would be e.g.:
            response = requests.post(
                DOWNSTREAM_URL,
                json=ctx.payload,
                timeout=(self._connect_timeout_sec, self._read_timeout_sec),  # requests
                                                                               # natively
                                                                               # supports
                                                                               # (connect, read)
            )
            if response.status_code >= 500:
                raise DependencyError("Downstream 5xx", retryable=True, status_code=response.status_code)
            if response.status_code == 429:
                raise DependencyError("Downstream rate limited us", retryable=True, status_code=429)
            if response.status_code >= 400:
                raise DependencyError("Downstream 4xx (bad request)", retryable=False, status_code=response.status_code)
            return response.json()
        """
        if self._simulated_hang_rate and random.random() < self._simulated_hang_rate:
            # Simulate a dependency that accepts the call but never responds --
            # this is exactly the scenario the enforced timeout below protects against.
            time.sleep(self._read_timeout_sec + 30)  # "hangs" well past our timeout

        if random.random() < self._simulated_failure_rate:
            raise DependencyError(
                "Simulated transient downstream failure (5xx)",
                retryable=True,
                status_code=503,
            )

        return {"status": "ok", "echo": ctx.payload, "processed_by": "dependency_service"}

    def call(self, ctx: RequestContext) -> Dict[str, Any]:
        total_timeout = self._connect_timeout_sec + self._read_timeout_sec

        future = self._executor.submit(self._do_call, ctx)
        try:
            return future.result(timeout=total_timeout)
        except concurrent.futures.TimeoutError:
            # Enforced wall-clock timeout tripped -- the underlying call is abandoned
            # (the thread will finish in the background and its result discarded; a real
            # HTTP client would additionally need its OWN internal timeout/cancellation
            # so the abandoned connection is actually closed rather than just ignored here).
            logger.warning(
                "[%s] Dependency call exceeded enforced timeout of %.1fs -- treating as failure",
                ctx.request_id, total_timeout,
            )
            raise DependencyError(
                f"Dependency call timed out after {total_timeout:.1f}s "
                f"(connect={self._connect_timeout_sec}s, read={self._read_timeout_sec}s)",
                retryable=True,
                status_code=None,
            )


# ========================================================================================
# 10. ORCHESTRATOR -- wires all stages together exactly per the flow diagram
# ========================================================================================

class RequestPipeline:
    def __init__(
        self,
        authenticator: Authenticator,
        authorizer: Authorizer,
        idempotency_checker: IdempotencyChecker,
        idempotency_store: IdempotencyStore,
        rate_limiter_guard: RateLimiterGuard,
        dependency_client: DependencyClient,
        circuit_breaker: CircuitBreaker,
        max_retries: int = 3,
        required_scope: str = "read:resource",
    ):
        self._authenticator = authenticator
        self._authorizer = authorizer
        self._idempotency_checker = idempotency_checker
        self._idempotency_store = idempotency_store
        self._rate_limiter_guard = rate_limiter_guard
        self._dependency_client = dependency_client
        self._circuit_breaker = circuit_breaker
        self._max_retries = max_retries
        self._required_scope = required_scope

    # ------------------------------------------------------------------------------
    # Dependency call wrapped with circuit breaker + retry/backoff.
    # This is the box in the diagram that loops: Dependency Call -> Failure ->
    # retryable? -> Backoff -> Retry -> Circuit Breaker -> (back to Dependency Call).
    # ------------------------------------------------------------------------------
    def _call_dependency_with_resilience(self, ctx: RequestContext) -> Dict[str, Any]:
        last_error: Optional[Exception] = None

        for attempt in range(self._max_retries + 1):  # +1 to include the initial attempt
            ctx.attempt = attempt

            # --- Circuit breaker gate: check BEFORE attempting the call ---
            if not self._circuit_breaker.allow_request():
                logger.warning(
                    "[%s] Circuit breaker is OPEN -- failing fast without calling dependency",
                    ctx.request_id,
                )
                raise CircuitOpenError("Circuit breaker is open; dependency presumed unhealthy")

            try:
                logger.info(
                    "[%s] Calling dependency (attempt %d/%d)...",
                    ctx.request_id, attempt + 1, self._max_retries + 1,
                )
                result = self._dependency_client.call(ctx)

                # Success -> tell the circuit breaker so it can heal/reset.
                self._circuit_breaker.record_success()
                logger.info("[%s] Dependency call SUCCEEDED on attempt %d", ctx.request_id, attempt + 1)
                return result

            except DependencyError as exc:
                last_error = exc
                self._circuit_breaker.record_failure()

                if not exc.retryable:
                    logger.error(
                        "[%s] Dependency call FAILED with NON-RETRYABLE error: %s",
                        ctx.request_id, exc,
                    )
                    raise  # -> "NO" branch in the diagram: Fail immediately.

                if attempt >= self._max_retries:
                    logger.error(
                        "[%s] Dependency call FAILED after exhausting %d retries: %s",
                        ctx.request_id, self._max_retries, exc,
                    )
                    break  # exit loop, will raise last_error below

                # -> "YES" branch in the diagram: retryable failure, so back off and retry.
                delay = compute_backoff_delay(attempt)
                logger.warning(
                    "[%s] Dependency call FAILED (retryable): %s. "
                    "Backing off %.2fs before retry %d/%d...",
                    ctx.request_id, exc, delay, attempt + 1, self._max_retries,
                )
                time.sleep(delay)
                # loop continues -> retries the dependency call

        # Retries exhausted for a retryable error.
        raise DependencyError(
            f"Dependency call failed after {self._max_retries} retries: {last_error}",
            retryable=True,
        )

    # ------------------------------------------------------------------------------
    # Full pipeline, top to bottom, exactly matching the flow diagram's ordering:
    # Auth -> Authz -> Idempotency -> Rate Limit -> Dependency Call -> Final Result
    # ------------------------------------------------------------------------------
    def handle_request(self, ctx: RequestContext) -> Dict[str, Any]:
        logger.info("=" * 80)
        logger.info("[%s] --> New request received", ctx.request_id)

        # 1. Authentication
        self._authenticator.authenticate(ctx)

        # 2. Authorization
        self._authorizer.authorize(ctx, required_scope=self._required_scope)

        # 3. Idempotency check (may short-circuit with a cached result)
        cached_result = self._idempotency_checker.check(ctx)
        if cached_result is not None:
            logger.info("[%s] <-- Returning cached (idempotent) result", ctx.request_id)
            return cached_result

        # 4. Rate limiting
        self._rate_limiter_guard.check(ctx)

        # 5. Dependency call (with retry/backoff/circuit-breaker wrapped around it)
        try:
            result = self._call_dependency_with_resilience(ctx)
        except PipelineError:
            # If this request had an idempotency key, release the lock on failure so a
            # legitimate future retry with the same key isn't stuck behind a phantom
            # IN_PROGRESS entry forever.
            if ctx.idempotency_key is not None:
                self._idempotency_store.release_on_failure(ctx.idempotency_key)
            raise

        # If we got here, the dependency call succeeded -> record for idempotency.
        if ctx.idempotency_key is not None:
            self._idempotency_store.mark_completed(ctx.idempotency_key, result)

        logger.info("[%s] <-- Request completed successfully", ctx.request_id)
        return result


# ========================================================================================
# 11. DEMO / WIRING -- shows how you'd assemble and exercise this pipeline
# ========================================================================================

def build_demo_pipeline(
    simulated_failure_rate: float = 0.5,
    simulated_hang_rate: float = 0.0,
) -> RequestPipeline:
    authenticator = Authenticator(valid_tokens={"valid-token-abc": "user-123"})

    authorizer = Authorizer(entitlements={"user-123": ["read:resource", "write:resource"]})

    idempotency_store = IdempotencyStore()
    idempotency_checker = IdempotencyChecker(store=idempotency_store)

    rate_limiter = TokenBucketRateLimiter(capacity=5, refill_rate_per_sec=1.0)
    rate_limiter_guard = RateLimiterGuard(limiter=rate_limiter)

    dependency_client = DependencyClient(
        simulated_failure_rate=simulated_failure_rate,
        simulated_hang_rate=simulated_hang_rate,
        connect_timeout_sec=0.5,   # kept short here so the demo runs quickly
        read_timeout_sec=1.0,
    )

    circuit_breaker = CircuitBreaker(
        failure_threshold=3,
        failure_window_sec=30.0,
        reset_timeout_sec=5.0,
        reset_timeout_jitter_sec=2.0,
        half_open_trial_calls=1,
        success_threshold_half_open=2,
    )

    return RequestPipeline(
        authenticator=authenticator,
        authorizer=authorizer,
        idempotency_checker=idempotency_checker,
        idempotency_store=idempotency_store,
        rate_limiter_guard=rate_limiter_guard,
        dependency_client=dependency_client,
        circuit_breaker=circuit_breaker,
        max_retries=3,
        required_scope="read:resource",
    )


if __name__ == "__main__":
    pipeline = build_demo_pipeline(simulated_failure_rate=0.6)

    # Example 1: a normal request
    ctx1 = RequestContext(
        request_id=str(uuid.uuid4())[:8],
        caller_token="valid-token-abc",
        idempotency_key="order-42-create",
        payload={"order_id": 42, "action": "create"},
    )
    try:
        result = pipeline.handle_request(ctx1)
        print("\nFINAL RESULT (request 1):", result)
    except PipelineError as e:
        print("\nREQUEST 1 FAILED:", e)

    # Example 2: replay the SAME idempotency key -- should hit the idempotency cache
    # if request 1 succeeded, demonstrating no duplicate dependency call happens.
    ctx2 = RequestContext(
        request_id=str(uuid.uuid4())[:8],
        caller_token="valid-token-abc",
        idempotency_key="order-42-create",  # same key as ctx1
        payload={"order_id": 42, "action": "create"},
    )
    try:
        result = pipeline.handle_request(ctx2)
        print("\nFINAL RESULT (request 2, replayed idempotency key):", result)
    except PipelineError as e:
        print("\nREQUEST 2 FAILED:", e)

    # Example 3: a caller with an invalid token -- should fail at Authentication.
    ctx3 = RequestContext(
        request_id=str(uuid.uuid4())[:8],
        caller_token="bad-token",
        payload={"order_id": 99},
    )
    try:
        result = pipeline.handle_request(ctx3)
        print("\nFINAL RESULT (request 3):", result)
    except PipelineError as e:
        print("\nREQUEST 3 FAILED (expected -- bad token):", e)
