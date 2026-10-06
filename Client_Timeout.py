class DependencyClient:
    def __init__(self, simulated_failure_rate: float = 0.0):
        self._simulated_failure_rate = simulated_failure_rate

    def call(self, ctx: RequestContext) -> Dict[str, Any]:
        """
        Real implementation would be e.g.:
            response = requests.post(DOWNSTREAM_URL, json=ctx.payload, timeout=5)
            ...
        """
        # NOTE: no timeout enforcement exists ANYWHERE in this method.
        # A hung call simply never returns and never raises.
        if random.random() < self._simulated_failure_rate:
            raise DependencyError(
                "Simulated transient downstream failure (timeout/503)",
                retryable=True,
                status_code=503,
            )
        return {"status": "ok", "echo": ctx.payload, "processed_by": "dependency_service"}