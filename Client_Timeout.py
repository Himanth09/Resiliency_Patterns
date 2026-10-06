class DependencyClient:
    def __init__(
        self,
        simulated_failure_rate: float = 0.0,
        simulated_hang_rate: float = 0.0,
        connect_timeout_sec: float = 2.0,   # NEW: separate connect timeout
        read_timeout_sec: float = 5.0,      # NEW: separate read timeout
    ):
        self._simulated_failure_rate = simulated_failure_rate
        self._simulated_hang_rate = simulated_hang_rate
        self._connect_timeout_sec = connect_timeout_sec
        self._read_timeout_sec = read_timeout_sec
        # Bounded pool -- avoids unbounded thread creation under concurrent load.
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=20, thread_name_prefix="dependency-call"
        )

    def _do_call(self, ctx: RequestContext) -> Dict[str, Any]:
        """
        Real implementation would be e.g.:
            response = requests.post(
                DOWNSTREAM_URL, json=ctx.payload,
                timeout=(self._connect_timeout_sec, self._read_timeout_sec),
            )
            ...
        """
        if self._simulated_hang_rate and random.random() < self._simulated_hang_rate:
            time.sleep(self._read_timeout_sec + 30)  # simulate a hang

        if random.random() < self._simulated_failure_rate:
            raise DependencyError("Simulated transient downstream failure (5xx)",
                                   retryable=True, status_code=503)
        return {"status": "ok", "echo": ctx.payload, "processed_by": "dependency_service"}

    def call(self, ctx: RequestContext) -> Dict[str, Any]:
        total_timeout = self._connect_timeout_sec + self._read_timeout_sec

        # ENFORCED wall-clock timeout, independent of whatever the underlying
        # HTTP client/SDK does internally -- defense-in-depth against a misconfigured
        # or misbehaving client that ignores its own timeout parameter.
        future = self._executor.submit(self._do_call, ctx)
        try:
            return future.result(timeout=total_timeout)
        except concurrent.futures.TimeoutError:
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
