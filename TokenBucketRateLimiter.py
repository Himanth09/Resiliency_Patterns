 def __init__(self, capacity: int, refill_rate_per_sec: float):
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
            # Double-checked locking to avoid a race on first access.
            bucket = self._buckets.get(caller_id)
            if bucket is None:
                bucket = {
                    "tokens": float(self._capacity),
                    "last_refill": time.monotonic(),
                    "lock": threading.Lock(),  # PER-CALLER lock, not global
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
        """Seconds until a token is available -- used for a precise Retry-After."""
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