"""Observe-Decide-Act routing engine."""

import asyncio
import time
from collections.abc import Mapping

from adapters.registry import AdapterRegistry
from observability.logging import get_logger
from observability.metrics import (
    provider_error_rate,
    router_cost_usd_total,
    router_latency_seconds,
    router_requests_total,
)
from router.analyzer import RequestAnalyzer
from router.audit import AuditLog
from router.config import RouterSettings, default_model_catalog
from router.schemas import (
    AuditRecord,
    ModelCandidate,
    ProviderResponse,
    RouterRequest,
    RouterResponse,
)
from router.state import RequestState, RoutingStateMachine
from router.strategies import (
    InflightStats,
    LatencyStats,
    RoutingStrategy,
    SuccessStats,
    build_strategies,
)
from safety.budget import BudgetExceededError, BudgetGuardrail
from safety.circuit_breaker import CircuitBreakerRegistry, CircuitOpenError
from safety.pii import PiiScrubber
from safety.rate_limiter import TokenBucketRateLimiter


class RoutingFailedError(RuntimeError):
    """Raised when no provider in the fallback chain succeeds."""


class NexusRouter:
    """Production routing engine implementing Observe, Decide, and Act."""

    def __init__(
        self,
        settings: RouterSettings,
        adapter_registry: AdapterRegistry,
        analyzer: RequestAnalyzer | None = None,
        model_catalog: Mapping[str, ModelCandidate] | None = None,
    ) -> None:
        """Initialize router dependencies.

        Args:
            settings: Router runtime settings.
            adapter_registry: Provider adapter registry.
            analyzer: Optional request analyzer override.
            model_catalog: Optional model catalog override.
        """
        self._settings = settings
        self._adapter_registry = adapter_registry
        self._analyzer = analyzer or RequestAnalyzer()
        self._model_catalog = dict(model_catalog or default_model_catalog())
        self._latency_stats = LatencyStats()
        self._inflight_stats = InflightStats()
        self._success_stats = SuccessStats()
        self._circuit_breakers = CircuitBreakerRegistry()
        self._strategies = build_strategies(
            self._model_catalog,
            self._latency_stats,
            self._inflight_stats,
            settings.quality_floor,
            settings.ab_model_a,
            settings.ab_model_b,
            settings.ab_model_a_weight,
            self._circuit_breakers,
            settings.blend_quality_weight,
            settings.blend_cost_weight,
            settings.blend_latency_weight,
            settings.request_cost_ceiling_usd,
            settings.canary_stable_model,
            settings.canary_model,
            settings.canary_weight,
            settings.latency_sla_ms,
            settings.epsilon,
            settings.availability_slo,
            self._success_stats,
            settings.failover_priority,
            settings.health_blend_success_weight,
            settings.health_blend_latency_weight,
            settings.health_blend_quality_weight,
            settings.health_blend_cost_weight,
        )
        self._audit_log = AuditLog(settings.audit_log_path)
        self._budget_guardrail = BudgetGuardrail(settings.budget_cap_usd)
        self._pii_scrubber = PiiScrubber(settings.enable_pii_scrubbing)
        self._rate_limiter = TokenBucketRateLimiter(
            settings.rate_limit_capacity,
            settings.rate_limit_refill_per_second,
        )
        self._logger = get_logger(__name__)

    async def complete(self, request: RouterRequest) -> RouterResponse:
        """Route and dispatch a completion request.

        Args:
            request: Router request.

        Returns:
            Router response.

        Raises:
            RoutingFailedError: If all provider attempts fail.
        """
        state_machine = RoutingStateMachine()
        started_at = time.perf_counter()
        self._rate_limiter.assert_allowed(request.api_key_id)
        strategy_name = request.strategy or self._settings.default_strategy
        router_requests_total.labels(strategy_name.value, state_machine.current_state.value).inc()

        signals = self._analyzer.analyze(request)
        state_machine.transition(RequestState.CLASSIFIED)
        strategy = self._strategies[strategy_name]
        decision = strategy.choose(request, signals)
        state_machine.transition(RequestState.ROUTED)
        self._logger.info(
            "routing_decision",
            request_id=request.request_id,
            model=decision.chosen_model,
            strategy=decision.routing_strategy.value,
            rationale=decision.rationale,
        )
        attempts = [decision.chosen_model, *decision.fallback_chain]
        last_error: Exception | None = None

        for attempt_index, model_name in enumerate(attempts):
            candidate = self._model_catalog[model_name]
            is_last_attempt = attempt_index == len(attempts) - 1

            # Client-side guardrails (budget cap, open circuit) are not provider
            # faults. Skipping a candidate for these reasons must not record a
            # provider failure or trip its circuit breaker, otherwise repeated
            # budget rejections could open a healthy provider's circuit and
            # degrade unrelated traffic.
            try:
                estimated_cost = candidate.estimate_cost(
                    signals.prompt_tokens_estimate, request.max_tokens
                )
                self._budget_guardrail.assert_can_spend(request.user_id, estimated_cost)
                self._circuit_breakers.assert_available(candidate.provider)
            except (BudgetExceededError, CircuitOpenError) as guardrail_error:
                last_error = guardrail_error
                self._logger.warning(
                    "provider_attempt_skipped",
                    request_id=request.request_id,
                    provider=candidate.provider,
                    model=model_name,
                    error=str(guardrail_error),
                )
                if is_last_attempt:
                    state_machine.transition(RequestState.FAILED)
                continue

            try:
                adapter = self._adapter_registry.get(candidate.provider)
                dispatchable_state = state_machine.current_state in {
                    RequestState.ROUTED,
                    RequestState.FALLBACK,
                }
                if dispatchable_state:
                    state_machine.transition(RequestState.DISPATCHED)
                sanitized_messages = self._pii_scrubber.scrub_messages(request.messages)
                self._inflight_stats.begin(candidate.provider)
                try:
                    provider_response = await asyncio.wait_for(
                        adapter.complete(model_name, sanitized_messages, request.max_tokens),
                        timeout=self._settings.provider_settings.request_timeout_seconds,
                    )
                finally:
                    self._inflight_stats.finish(candidate.provider)
                return self._respond(
                    request=request,
                    provider_response=provider_response,
                    strategy=strategy,
                    rationale=self._rationale(decision.rationale, model_name, attempt_index),
                    state_machine=state_machine,
                    started_at=started_at,
                    provider=candidate.provider,
                )
            except Exception as exception:
                last_error = exception
                self._circuit_breakers.record_failure(candidate.provider)
                self._success_stats.observe(candidate.provider, success=False)
                provider_error_rate.labels(candidate.provider, model_name).inc()
                self._logger.warning(
                    "provider_attempt_failed",
                    request_id=request.request_id,
                    provider=candidate.provider,
                    model=model_name,
                    error=str(exception),
                )
                if not is_last_attempt:
                    if state_machine.current_state is RequestState.DISPATCHED:
                        state_machine.transition(RequestState.FALLBACK)
                    continue
                state_machine.transition(RequestState.FAILED)

        message = f"all provider attempts failed: {last_error}"
        raise RoutingFailedError(message)

    def _respond(
        self,
        request: RouterRequest,
        provider_response: ProviderResponse,
        strategy: RoutingStrategy,
        rationale: str,
        state_machine: RoutingStateMachine,
        started_at: float,
        provider: str,
    ) -> RouterResponse:
        """Build the router response and persist observability side effects."""
        self._circuit_breakers.record_success(provider)
        self._success_stats.observe(provider, success=True)
        state_machine.transition(RequestState.RESPONDED)
        latency_ms = (time.perf_counter() - started_at) * 1000.0
        latency_seconds = latency_ms / 1000.0
        self._latency_stats.observe(provider, latency_ms)
        self._budget_guardrail.record_spend(request.user_id, provider_response.cost_usd)
        router_cost_usd_total.labels(provider, provider_response.model).inc(
            provider_response.cost_usd
        )
        router_latency_seconds.labels(strategy.strategy_name.value, provider).observe(
            latency_seconds
        )
        router_requests_total.labels(
            strategy.strategy_name.value, state_machine.current_state.value
        ).inc()
        response = RouterResponse(
            content=provider_response.content,
            model_used=provider_response.model,
            routing_strategy=strategy.strategy_name,
            latency_ms=latency_ms,
            input_tokens=provider_response.input_tokens,
            output_tokens=provider_response.output_tokens,
            cost_usd=provider_response.cost_usd,
            rationale=rationale,
            request_id=request.request_id,
        )
        self._audit_log.append(
            AuditRecord(
                request_id=request.request_id,
                chosen_model=provider_response.model,
                provider=provider,
                routing_strategy=strategy.strategy_name,
                rationale=rationale,
                latency_ms=latency_ms,
                token_cost=provider_response.cost_usd,
                input_tokens=provider_response.input_tokens,
                output_tokens=provider_response.output_tokens,
                state=state_machine.current_state.value,
            ),
        )
        self._logger.info(
            "routing_response",
            request_id=request.request_id,
            model=response.model_used,
            latency_ms=response.latency_ms,
            cost_usd=response.cost_usd,
        )
        return response

    @staticmethod
    def _rationale(base_rationale: str, model_name: str, attempt_index: int) -> str:
        """Return rationale with fallback context.

        Args:
            base_rationale: Original strategy rationale.
            model_name: Model that produced the response.
            attempt_index: Attempt index in the fallback chain.

        Returns:
            Rationale text.
        """
        if attempt_index == 0:
            return base_rationale
        return f"{base_rationale}; fallback attempt {attempt_index} succeeded with {model_name}"
