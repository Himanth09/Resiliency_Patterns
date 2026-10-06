class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, reset_timeout_sec: float = 30.0,
                 half_open_trial_calls: int = 1):
        self._failure_threshold = failure_threshold
        self._reset_timeout_sec = reset_timeout_sec
        self._half_open_trial_calls = half_open_trial_calls
        self._state = CircuitState.CLOSED
        self._failure_count = 0  # simple consecutive counter
        self._opened_at: Optional[float] = None
        self._half_open_calls_in_flight = 0
        self._lock = threading.Lock()

    def _maybe_transition_to_half_open(self) -> None:
        if self._state == CircuitState.OPEN and self._opened_at is not None:
            if (time.monotonic() - self._opened_at) >= self._reset_timeout_sec:  # fixed, no jitter
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls_in_flight = 0

    def allow_request(self) -> bool:
        with self._lock:
            self._maybe_transition_to_half_open()
            if self._state == CircuitState.CLOSED:
                return True
            if self._state == CircuitState.OPEN:
                return False
            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls_in_flight < self._half_open_trial_calls:
                    self._half_open_calls_in_flight += 1
                    return True
                return False
            return False

    def record_success(self) -> None:
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED  # ONE success closes it immediately
                self._failure_count = 0
                self._opened_at = None
                self._half_open_calls_in_flight = 0
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0  # any success wipes the whole counter

    def record_failure(self) -> None:
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
                self._half_open_calls_in_flight = 0
                return
            self._failure_count += 1
            if self._failure_count >= self._failure_threshold:
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            self._maybe_transition_to_half_open()
            return self._state