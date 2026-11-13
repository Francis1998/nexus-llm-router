"""Observe-Decide-Act routing engine."""

import asyncio
import sqlite3
import time
from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from types import MappingProxyType

from adapters.base import ProviderError
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
from router.fallback_scoreboard import ProviderFallbackScoreboard
from router.feedback import FeedbackStore
from router.schemas import (
    AuditRecord,
    DomainTag,
    ModelCandidate,
    ProviderResponse,
    RouterRequest,
    RouterResponse,
)
from router.state import RequestState, RoutingStateMachine
from router.strategies import (
    CanaryShadowSplitStats,
    CostAnomalyStats,
    FamilySpendWindow,
    InflightStats,
    LatencySlopeStats,
    LatencyStats,
    ProviderColdStartStats,
    ProviderErrorBudgetResetStats,
    ProviderHourlySpendWindow,
    ProviderRetryAfterCooldown,
    ProviderWeightStats,
    RateLimitStats,
    RegionFailoverHysteresisStats,
    RoutingStrategy,
    StickyModelPinExpireStats,
    StickyRegionDrainStats,
    StickyRegionWarmupStats,
    StickySessionMigrateStats,
    SuccessStats,
    TenantBudgetCascadeStats,
    TenantBudgetCascadeStrategy,
    TenantFairQueueStats,
    TenantPriorityLaneStats,
    TenantQuotaBurstStats,
    TierRequestStats,
    TokenBucketStats,
    TokenRpmWindow,
    build_strategies,
)
from safety.budget import BudgetExceededError, BudgetGuardrail
from safety.circuit_breaker import CircuitBreakerRegistry, CircuitOpenError
from safety.pii import PiiScrubber
from safety.rate_limiter import TokenBucketRateLimiter
from safety.spend_ledger import SpendLedger

# Provider HTTP statuses that reject the request itself (malformed input, a
# prompt too large for the model). They are the caller's fault, unlike 401/403/404
# (gateway credentials or model id), 408/409 (transient), 429 (quota), and 5xx.
_CLIENT_ERROR_STATUS_CODES = frozenset({400, 413, 422})


class RoutingFailedError(RuntimeError):
    """Raised when no provider in the fallback chain succeeds."""


class ProviderRejectedRequestError(RoutingFailedError):
    """Raised when a provider rejects the request itself (HTTP 400, 413, or 422).

    The request, not the provider, is at fault, so the rejection is neither
    recorded against provider health nor retried on another provider.
    """

    def __init__(self, message: str, *, status_code: int) -> None:
        """Initialize the error.

        Args:
            message: Human-readable rejection description.
            status_code: Provider HTTP status to return to the caller.
        """
        super().__init__(message)
        self.status_code = status_code


class UnpricedModelError(RoutingFailedError):
    """Raised for a model that neither its adapter nor the catalog prices.

    Its spend could not be recorded or held against a budget, so the attempt is
    skipped before dispatch (fail closed) and the fallback chain continues. The
    router raises it for the request only when no candidate has a price: a
    gateway configuration error, not a budget overrun.
    """


class BudgetRefusedError(RoutingFailedError, BudgetExceededError):
    """Raised when the budget guardrail refused every candidate with a price.

    No candidate fit the subject's remaining budget, so retrying cannot succeed
    until the budget allows it. It is a ``BudgetExceededError`` for the API's
    402 mapping and a ``RoutingFailedError`` for callers catching that.
    """


class NexusRouter:
    """Production routing engine implementing Observe, Decide, and Act."""

    def __init__(
        self,
        settings: RouterSettings,
        adapter_registry: AdapterRegistry,
        analyzer: RequestAnalyzer | None = None,
        model_catalog: Mapping[str, ModelCandidate] | None = None,
        feedback_store: FeedbackStore | None = None,
    ) -> None:
        """Initialize router dependencies.

        Args:
            settings: Router runtime settings.
            adapter_registry: Provider adapter registry.
            analyzer: Optional request analyzer override.
            model_catalog: Optional model catalog override.
            feedback_store: Optional request-feedback store override; defaults
                to one at ``settings.feedback_path``.
        """
        self._settings = settings
        self._adapter_registry = adapter_registry
        self._analyzer = analyzer or RequestAnalyzer()
        self._model_catalog = dict(model_catalog or default_model_catalog())
        self._budget_guardrail = BudgetGuardrail(settings.budget_cap_usd)
        self._feedback_store = feedback_store or FeedbackStore(
            settings.feedback_path, window_seconds=settings.feedback_window_seconds
        )
        self._latency_stats = LatencyStats()
        self._inflight_stats = InflightStats()
        self._success_stats = SuccessStats()
        self._rate_limit_stats = RateLimitStats()
        self._token_bucket_stats = TokenBucketStats(
            settings.token_bucket_capacity,
            settings.token_bucket_refill_per_sec,
        )
        self._tier_request_stats = TierRequestStats()
        self._family_spend_window = FamilySpendWindow(
            settings.soft_family_budget_window_seconds,
        )
        self._cost_anomaly_stats = CostAnomalyStats()
        self._provider_weight_stats = ProviderWeightStats(
            settings.provider_weight_decay_factor,
            settings.provider_weight_recover,
        )
        self._retry_after_cooldown = ProviderRetryAfterCooldown(
            settings.retry_after_default_seconds
        )
        self._latency_slope_stats = LatencySlopeStats(settings.latency_slope_window)
        self._provider_hourly_spend_window = ProviderHourlySpendWindow()
        self._token_rpm_window = TokenRpmWindow()
        self._region_failover_hysteresis_stats = RegionFailoverHysteresisStats()
        self._tenant_budget_cascade_stats = TenantBudgetCascadeStats()
        self._provider_error_budget_reset_stats = ProviderErrorBudgetResetStats(
            settings.provider_error_budget_reset_seconds
        )
        self._sticky_region_warmup_stats = StickyRegionWarmupStats()
        self._tenant_quota_burst_stats = TenantQuotaBurstStats(
            settings.tenant_quota_burst_window_seconds
        )
        self._sticky_session_migrate_stats = StickySessionMigrateStats()
        self._provider_cold_start_stats = ProviderColdStartStats(
            settings.provider_cold_start_lookback
        )
        self._tenant_fair_queue_stats = TenantFairQueueStats(settings.tenant_fair_queue_lookback)
        self._sticky_region_drain_stats = StickyRegionDrainStats()
        self._provider_canary_shadow_stats = CanaryShadowSplitStats()
        self._sticky_model_pin_expire_stats = StickyModelPinExpireStats()
        self._tenant_priority_lane_stats = TenantPriorityLaneStats(
            settings.tenant_priority_lane_lookback
        )
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
            settings.prompt_prefix_cache_min_chars,
            settings.epsilon,
            settings.availability_slo,
            self._success_stats,
            settings.failover_priority,
            settings.health_blend_success_weight,
            settings.health_blend_latency_weight,
            settings.health_blend_quality_weight,
            settings.health_blend_cost_weight,
            settings.concurrency_cap,
            rate_limit_stats=self._rate_limit_stats,
            token_bucket_stats=self._token_bucket_stats,
            hcl_health_weight=settings.hcl_health_weight,
            hcl_cost_weight=settings.hcl_cost_weight,
            hcl_latency_weight=settings.hcl_latency_weight,
            tier_request_stats=self._tier_request_stats,
            tier_frontier_rpm=settings.tier_frontier_rpm,
            tier_mid_rpm=settings.tier_mid_rpm,
            tier_economy_rpm=settings.tier_economy_rpm,
            provider_family_cost_ceiling_usd=settings.provider_family_cost_ceiling_usd,
            adaptive_exploration_base=settings.adaptive_exploration_base,
            adaptive_exploration_min=settings.adaptive_exploration_min,
            family_spend_window=self._family_spend_window,
            soft_family_budget_usd=settings.soft_family_budget_usd,
            sticky_region_failover_preferences=settings.sticky_region_failover_preferences,
            latency_slo_ms=settings.latency_slo_ms,
            shadow_traffic_percent=settings.shadow_traffic_percent,
            canary_cost_blend_percent=settings.canary_cost_blend_percent,
            cost_anomaly_stats=self._cost_anomaly_stats,
            token_cost_anomaly_ratio=settings.token_cost_anomaly_ratio,
            latency_hedge_ms=settings.latency_hedge_ms,
            prompt_length_tier_tokens=settings.prompt_length_tier_tokens,
            retry_budget_default=settings.retry_budget_default,
            cache_hit_sticky_min_chars=settings.cache_hit_sticky_min_chars,
            embedding_cache_namespace_prefix=settings.embedding_cache_namespace_prefix,
            circuit_half_open_probe_budget=settings.circuit_half_open_probe_budget,
            semantic_cache_ttl_seconds=settings.semantic_cache_ttl_seconds,
            provider_spend_soft_usd=settings.provider_spend_soft_usd,
            carbon_aware_max_intensity=settings.carbon_aware_max_intensity,
            tenant_concurrency_lease=settings.tenant_concurrency_lease,
            provider_error_budget_rate=settings.provider_error_budget_rate,
            region_latency_p99_ms=settings.region_latency_p99_ms,
            sticky_canary_cost_percent=settings.sticky_canary_cost_percent,
            queue_depth_soft_cap=settings.queue_depth_soft_cap,
            provider_quota_lookback=settings.provider_quota_lookback,
            adaptive_timeout_hedge_ratio=settings.adaptive_timeout_hedge_ratio,
            token_bucket_tenant_rate=settings.token_bucket_tenant_rate,
            region_carbon_blend_weight=settings.region_carbon_blend_weight,
            provider_weight_decay_factor=settings.provider_weight_decay_factor,
            provider_weight_recover=settings.provider_weight_recover,
            provider_weight_stats=self._provider_weight_stats,
            retry_after_default_seconds=settings.retry_after_default_seconds,
            retry_after_cooldown=self._retry_after_cooldown,
            latency_slope_window=settings.latency_slope_window,
            latency_slope_threshold_ms=settings.latency_slope_threshold_ms,
            latency_slope_stats=self._latency_slope_stats,
            provider_hourly_cost_ceiling_usd=settings.provider_hourly_cost_ceiling_usd,
            provider_hourly_spend_window=self._provider_hourly_spend_window,
            token_rpm_ceiling=settings.token_rpm_ceiling,
            provider_circuit_probe_budget=settings.provider_circuit_probe_budget,
            carbon_latency_latency_weight=settings.carbon_latency_latency_weight,
            carbon_latency_carbon_weight=settings.carbon_latency_carbon_weight,
            token_rpm_window=self._token_rpm_window,
            adaptive_concurrency_base_cap=settings.adaptive_concurrency_base_cap,
            adaptive_concurrency_min_cap=settings.adaptive_concurrency_min_cap,
            adaptive_concurrency_latency_ms=settings.adaptive_concurrency_latency_ms,
            provider_token_fair_share_ceiling=settings.provider_token_fair_share_ceiling,
            region_failover_hysteresis_successes=settings.region_failover_hysteresis_successes,
            region_failover_hysteresis_stats=self._region_failover_hysteresis_stats,
            tenant_budget_cascade_soft=settings.tenant_budget_cascade_soft,
            tenant_budget_cascade_hard=settings.tenant_budget_cascade_hard,
            tenant_budget_cascade_stats=self._tenant_budget_cascade_stats,
            provider_error_budget_reset_fraction=settings.provider_error_budget_reset_fraction,
            provider_error_budget_reset_seconds=settings.provider_error_budget_reset_seconds,
            provider_error_budget_reset_stats=self._provider_error_budget_reset_stats,
            sticky_region_warmup_requests=settings.sticky_region_warmup_requests,
            sticky_region_warmup_stats=self._sticky_region_warmup_stats,
            tenant_quota_burst_stats=self._tenant_quota_burst_stats,
            tenant_quota_burst_soft=settings.tenant_quota_burst_soft,
            tenant_quota_burst_hard=settings.tenant_quota_burst_hard,
            tenant_quota_burst_window_seconds=settings.tenant_quota_burst_window_seconds,
            provider_tail_latency_hedge_ms=settings.provider_tail_latency_hedge_ms,
            sticky_session_migrate_stats=self._sticky_session_migrate_stats,
            sticky_session_migrate_success_threshold=(
                settings.sticky_session_migrate_success_threshold
            ),
            provider_cold_start_stats=self._provider_cold_start_stats,
            provider_cold_start_lookback=settings.provider_cold_start_lookback,
            provider_cold_start_target=settings.provider_cold_start_target,
            tenant_fair_queue_stats=self._tenant_fair_queue_stats,
            tenant_fair_queue_lookback=settings.tenant_fair_queue_lookback,
            sticky_region_drain_stats=self._sticky_region_drain_stats,
            sticky_region_drain_regions=settings.sticky_region_drain_regions,
            provider_canary_shadow_stats=self._provider_canary_shadow_stats,
            provider_canary_primary_provider=settings.provider_canary_primary_provider,
            provider_canary_shadow_percent=settings.provider_canary_shadow_percent,
            sticky_model_pin_expire_stats=self._sticky_model_pin_expire_stats,
            sticky_model_pin_ttl_seconds=settings.sticky_model_pin_ttl_seconds,
            tenant_priority_lane_stats=self._tenant_priority_lane_stats,
            tenant_priority_high_tenants=settings.tenant_priority_high_tenants,
            tenant_priority_low_tenants=settings.tenant_priority_low_tenants,
            tenant_priority_lane_lookback=settings.tenant_priority_lane_lookback,
            tenant_priority_high_quota=settings.tenant_priority_high_quota,
            tenant_priority_normal_quota=settings.tenant_priority_normal_quota,
            tenant_priority_low_quota=settings.tenant_priority_low_quota,
            deadline_aware_threshold_ms=settings.deadline_aware_threshold_ms,
            provider_success_floor=settings.provider_success_floor,
            provider_warmup_blend=settings.provider_warmup_blend,
            tenant_soft_isolation_rpm=settings.tenant_soft_isolation_rpm,
            prompt_injection_risk_threshold=settings.prompt_injection_risk_threshold,
            thinking_complexity_threshold=settings.thinking_complexity_threshold,
            feedback_stats=self._feedback_store,
            dispatch_guard=self,
            feedback_metric=settings.feedback_metric,
            feedback_min_samples=settings.feedback_min_samples,
        )
        self._audit_log = AuditLog(settings.audit_log_path)
        self._spend_ledger = SpendLedger(settings.spend_ledger_path)
        self._pii_scrubber = PiiScrubber(settings.enable_pii_scrubbing)
        self._rate_limiter = TokenBucketRateLimiter(
            settings.rate_limit_capacity,
            settings.rate_limit_refill_per_second,
        )
        self._fallback_scoreboard = ProviderFallbackScoreboard()
        self._logger = get_logger(__name__)
        self._catalog_priced_models, self._unpriced_models = self._classify_model_prices()

    @property
    def model_catalog(self) -> Mapping[str, ModelCandidate]:
        """Return the routable model catalog (read-only view)."""
        return MappingProxyType(self._model_catalog)

    @property
    def feedback_store(self) -> FeedbackStore:
        """Return the store that indexes completed requests and their feedback."""
        return self._feedback_store

    def allows_dispatch(self, model: str, subject: str, estimated_cost_usd: float) -> bool:
        """Return whether ``model`` would pass the pre-dispatch price and budget checks.

        Read-only twin of the attempt loop's ``_assert_priced`` and
        ``BudgetGuardrail.assert_can_spend``, so a strategy can avoid choosing a
        model the engine would skip.

        Args:
            model: Catalog model name.
            subject: Budget subject (``RouterRequest.user_id``).
            estimated_cost_usd: Estimated cost of the request on ``model``.

        Returns:
            ``False`` when the model has no known price or does not fit the budget.
        """
        if model in self._unpriced_models:
            return False
        return self._budget_guardrail.can_spend(subject, estimated_cost_usd)

    def provider_routable(self, provider: str) -> bool:
        """Return whether the provider's circuit breaker currently allows routing.

        Args:
            provider: Provider name such as ``openai``.

        Returns:
            ``False`` while the provider's circuit is open.
        """
        return self._circuit_breakers.is_available(provider)

    async def complete(self, request: RouterRequest) -> RouterResponse:
        """Route and dispatch a completion request.

        Args:
            request: Router request.

        Returns:
            Router response.

        Raises:
            ProviderRejectedRequestError: If a provider rejects the request itself
                (HTTP 400, 413, or 422); no fallback is attempted.
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
        attempts = self._order_attempts(decision.chosen_model, decision.fallback_chain)
        attempt_errors: list[Exception] = []

        for attempt_index, model_name in enumerate(attempts):
            candidate = self._model_catalog[model_name]
            is_last_attempt = attempt_index == len(attempts) - 1

            # Client-side guardrails (budget cap, open circuit) are not provider
            # faults. Skipping a candidate for these reasons must not record a
            # provider failure or trip its circuit breaker, otherwise repeated
            # budget rejections could open a healthy provider's circuit and
            # degrade unrelated traffic.
            try:
                self._assert_priced(model_name)
                estimated_cost = candidate.estimate_cost(
                    signals.prompt_tokens_estimate, request.max_tokens
                )
                self._budget_guardrail.assert_can_spend(request.user_id, estimated_cost)
                self._circuit_breakers.assert_available(candidate.provider)
            except (BudgetExceededError, CircuitOpenError, UnpricedModelError) as guardrail_error:
                attempt_errors.append(guardrail_error)
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

            attempt_started_at = time.perf_counter()
            try:
                adapter = self._adapter_registry.get(candidate.provider)
                dispatchable_state = state_machine.current_state in {
                    RequestState.ROUTED,
                    RequestState.FALLBACK,
                }
                if dispatchable_state:
                    state_machine.transition(RequestState.DISPATCHED)
                sanitized_messages = self._pii_scrubber.scrub_messages(request.messages)
                tenant_key = self._tenant_inflight_key(request)
                self._inflight_stats.begin(candidate.provider)
                self._inflight_stats.begin_for_tenant(tenant_key, candidate.provider)
                try:
                    provider_response = await asyncio.wait_for(
                        adapter.complete(model_name, sanitized_messages, request.max_tokens),
                        timeout=self._settings.provider_settings.request_timeout_seconds,
                    )
                finally:
                    self._inflight_stats.finish_for_tenant(tenant_key, candidate.provider)
                    self._inflight_stats.finish(candidate.provider)
                return self._respond(
                    request=request,
                    provider_response=self._apply_catalog_price(model_name, provider_response),
                    strategy=strategy,
                    rationale=self._rationale(decision.rationale, model_name, attempt_index),
                    state_machine=state_machine,
                    started_at=started_at,
                    provider=candidate.provider,
                    prompt_tokens_estimate=signals.prompt_tokens_estimate,
                    attempt_started_at=attempt_started_at,
                    domain=signals.domain_tag,
                )
            except Exception as exception:
                rejection = self._client_rejection(
                    exception, request, candidate.provider, model_name, state_machine
                )
                if rejection is not None:
                    raise rejection from exception
                attempt_errors.append(exception)
                attempt_latency_ms = (time.perf_counter() - attempt_started_at) * 1000.0
                self._fallback_scoreboard.record_outcome(
                    candidate.provider,
                    False,
                    attempt_latency_ms,
                )
                self._circuit_breakers.record_failure(candidate.provider)
                self._success_stats.observe(candidate.provider, success=False)
                self._provider_error_budget_reset_stats.observe(
                    candidate.provider,
                    success=False,
                )
                self._provider_weight_stats.observe(candidate.provider, success=False)
                region_label = (request.region or "global").strip().lower()
                self._region_failover_hysteresis_stats.record_failure(region_label)
                rate_limited = self._is_rate_limit_error(exception)
                self._rate_limit_stats.observe(
                    candidate.provider,
                    rate_limited=rate_limited,
                )
                if rate_limited:
                    self._retry_after_cooldown.set_cooldown(
                        candidate.provider,
                        self._extract_retry_after_seconds(exception),
                    )
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

        raise self._routing_failure(attempt_errors)

    async def complete_stream(
        self,
        request: RouterRequest,
        *,
        on_complete: Callable[[RouterResponse], None] | None = None,
    ) -> AsyncGenerator[tuple[str, str], None]:
        """Route a request and stream provider chunks.

        Args:
            request: Router request.
            on_complete: Optional callback receiving the final ``RouterResponse``
                (usage, cost, latency, rationale) once the stream finished
                successfully, so callers can report it after the last chunk.

        Yields:
            Pairs of ``(model_name, text_chunk)`` from the selected provider.
            A final ``(model_name, "")`` chunk is not emitted; callers should
            send the SSE terminal frame themselves after the iterator ends.

        Raises:
            ProviderRejectedRequestError: If a provider rejects the request itself
                (HTTP 400, 413, or 422); no fallback is attempted.
            RoutingFailedError: If all provider attempts fail, or if a provider
                fails after it already streamed text (falling back then would
                splice a second answer onto the partial one).
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
        attempts = self._order_attempts(decision.chosen_model, decision.fallback_chain)
        attempt_errors: list[Exception] = []

        for attempt_index, model_name in enumerate(attempts):
            candidate = self._model_catalog[model_name]
            is_last_attempt = attempt_index == len(attempts) - 1

            try:
                self._assert_priced(model_name)
                estimated_cost = candidate.estimate_cost(
                    signals.prompt_tokens_estimate, request.max_tokens
                )
                self._budget_guardrail.assert_can_spend(request.user_id, estimated_cost)
                self._circuit_breakers.assert_available(candidate.provider)
            except (BudgetExceededError, CircuitOpenError, UnpricedModelError) as guardrail_error:
                attempt_errors.append(guardrail_error)
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

            attempt_started_at = time.perf_counter()
            output_started = False
            try:
                adapter = self._adapter_registry.get(candidate.provider)
                dispatchable_state = state_machine.current_state in {
                    RequestState.ROUTED,
                    RequestState.FALLBACK,
                }
                if dispatchable_state:
                    state_machine.transition(RequestState.DISPATCHED)
                sanitized_messages = self._pii_scrubber.scrub_messages(request.messages)
                tenant_key = self._tenant_inflight_key(request)
                self._inflight_stats.begin(candidate.provider)
                self._inflight_stats.begin_for_tenant(tenant_key, candidate.provider)
                assembled = ""
                reported: ProviderResponse | None = None
                try:
                    stream = adapter.stream(model_name, sanitized_messages, request.max_tokens)
                    async for chunk in stream:
                        if isinstance(chunk, ProviderResponse):
                            reported = chunk
                            continue
                        assembled += chunk
                        output_started = output_started or bool(chunk)
                        yield model_name, chunk
                finally:
                    self._inflight_stats.finish_for_tenant(tenant_key, candidate.provider)
                    self._inflight_stats.finish(candidate.provider)
                if reported is not None:
                    # Provider-reported usage, cost, and stop reason for the text sent.
                    provider_response = reported.model_copy(
                        update={"content": assembled, "model": model_name}
                    )
                else:
                    output_tokens = max(1, len(assembled) // 4)
                    provider_response = ProviderResponse(
                        content=assembled,
                        model=model_name,
                        input_tokens=signals.prompt_tokens_estimate,
                        output_tokens=output_tokens,
                        cost_usd=adapter.estimate_cost(
                            model_name, signals.prompt_tokens_estimate, output_tokens
                        ),
                    )
                router_response = self._respond(
                    request=request,
                    provider_response=self._apply_catalog_price(model_name, provider_response),
                    strategy=strategy,
                    rationale=self._rationale(decision.rationale, model_name, attempt_index),
                    state_machine=state_machine,
                    started_at=started_at,
                    provider=candidate.provider,
                    prompt_tokens_estimate=signals.prompt_tokens_estimate,
                    attempt_started_at=attempt_started_at,
                    domain=signals.domain_tag,
                )
            except Exception as exception:
                rejection = self._client_rejection(
                    exception, request, candidate.provider, model_name, state_machine
                )
                if rejection is not None:
                    raise rejection from exception
                attempt_errors.append(exception)
                attempt_latency_ms = (time.perf_counter() - attempt_started_at) * 1000.0
                self._fallback_scoreboard.record_outcome(
                    candidate.provider,
                    False,
                    attempt_latency_ms,
                )
                self._circuit_breakers.record_failure(candidate.provider)
                self._success_stats.observe(candidate.provider, success=False)
                self._provider_error_budget_reset_stats.observe(
                    candidate.provider,
                    success=False,
                )
                self._provider_weight_stats.observe(candidate.provider, success=False)
                region_label = (request.region or "global").strip().lower()
                self._region_failover_hysteresis_stats.record_failure(region_label)
                rate_limited = self._is_rate_limit_error(exception)
                self._rate_limit_stats.observe(
                    candidate.provider,
                    rate_limited=rate_limited,
                )
                if rate_limited:
                    self._retry_after_cooldown.set_cooldown(
                        candidate.provider,
                        self._extract_retry_after_seconds(exception),
                    )
                provider_error_rate.labels(candidate.provider, model_name).inc()
                self._logger.warning(
                    "provider_attempt_failed",
                    request_id=request.request_id,
                    provider=candidate.provider,
                    model=model_name,
                    error=str(exception),
                )
                if output_started:
                    # The caller already received this attempt's text; retrying a
                    # fallback would splice a second answer onto the first.
                    if state_machine.current_state is RequestState.DISPATCHED:
                        state_machine.transition(RequestState.FAILED)
                    message = (
                        f"provider {candidate.provider} failed after streaming output began: "
                        f"{exception}"
                    )
                    raise RoutingFailedError(message) from exception
                if not is_last_attempt:
                    if state_machine.current_state is RequestState.DISPATCHED:
                        state_machine.transition(RequestState.FALLBACK)
                    continue
                state_machine.transition(RequestState.FAILED)
            else:
                if on_complete is not None:
                    on_complete(router_response)
                return

        raise self._routing_failure(attempt_errors)

    def _classify_model_prices(self) -> tuple[frozenset[str], frozenset[str]]:
        """Find catalog models that their provider adapter cannot price.

        Adapters keep their own price tables, and ``estimate_cost`` returns 0.0
        for a model missing from one, which would bill catalog-only SKUs
        (fine-tunes, newly released models) at $0. Those models are billed at
        their catalog prices instead (``_apply_catalog_price``). A model whose
        catalog rates are $0 too has no known price, so it is never dispatched
        (``_assert_priced``). Both cases are logged once per model.

        Returns:
            ``(catalog_priced, unpriced)`` catalog model names.
        """
        adapters = dict(self._adapter_registry.items())
        catalog_priced: set[str] = set()
        unpriced: set[str] = set()
        for model_name, candidate in self._model_catalog.items():
            adapter = adapters.get(candidate.provider)
            if adapter is None or adapter.has_price(model_name):
                continue
            if candidate.input_cost_per_1k > 0 or candidate.output_cost_per_1k > 0:
                catalog_priced.add(model_name)
                self._logger.info(
                    "model_price_from_catalog",
                    provider=candidate.provider,
                    model=model_name,
                    reason="provider adapter has no price for the model",
                )
            else:
                unpriced.add(model_name)
                self._logger.warning(
                    "model_price_unknown",
                    provider=candidate.provider,
                    model=model_name,
                    reason="no adapter price and $0 catalog prices; the model is not dispatched",
                )
        return frozenset(catalog_priced), frozenset(unpriced)

    def _assert_priced(self, model_name: str) -> None:
        """Refuse to dispatch a model whose spend nothing can price.

        Args:
            model_name: Catalog model about to be attempted.

        Raises:
            UnpricedModelError: If neither its adapter nor the catalog prices it.
        """
        if model_name in self._unpriced_models:
            raise UnpricedModelError(
                f"model {model_name} has no price: its provider adapter does not list it "
                "and its catalog input and output prices are $0"
            )

    def _apply_catalog_price(
        self, model_name: str, provider_response: ProviderResponse
    ) -> ProviderResponse:
        """Bill catalog prices when the provider adapter cannot price the model.

        Args:
            model_name: Catalog model that produced the response.
            provider_response: Normalized provider response.

        Returns:
            The response, re-priced from the catalog (long-prompt tier included)
            when the adapter has no price for ``model_name``; otherwise unchanged.
        """
        if model_name not in self._catalog_priced_models:
            return provider_response
        cost_usd = self._model_catalog[model_name].estimate_cost(
            provider_response.input_tokens, provider_response.output_tokens
        )
        return provider_response.model_copy(update={"cost_usd": cost_usd})

    @staticmethod
    def _routing_failure(attempt_errors: Sequence[Exception]) -> RoutingFailedError:
        """Return the error for a request that no attempt could serve.

        Candidates refused before dispatch for the budget or for a missing price
        never reached a provider. When those refusals cover every attempt, the
        request fails for that reason: ``BudgetRefusedError`` (HTTP 402) when the
        budget refused at least one priced candidate, else ``UnpricedModelError``
        (HTTP 500). An open circuit or a provider failure means an affordable,
        priced candidate could not be served, which a retry may get past, so the
        request fails with ``RoutingFailedError`` (HTTP 503).

        Args:
            attempt_errors: Why each attempt was skipped or failed, in order.

        Returns:
            The error to raise.
        """
        last_error = attempt_errors[-1] if attempt_errors else None
        budget_refusals = [
            error for error in attempt_errors if isinstance(error, BudgetExceededError)
        ]
        refused_before_dispatch = all(
            isinstance(error, (BudgetExceededError, UnpricedModelError)) for error in attempt_errors
        )
        if attempt_errors and refused_before_dispatch:
            if budget_refusals:
                return BudgetRefusedError(str(budget_refusals[-1]))
            return UnpricedModelError(str(last_error))
        return RoutingFailedError(f"all provider attempts failed: {last_error}")

    def _respond(
        self,
        request: RouterRequest,
        provider_response: ProviderResponse,
        strategy: RoutingStrategy,
        rationale: str,
        state_machine: RoutingStateMachine,
        started_at: float,
        provider: str,
        prompt_tokens_estimate: int,
        attempt_started_at: float | None = None,
        domain: DomainTag | None = None,
    ) -> RouterResponse:
        """Build the router response and persist observability side effects."""
        self._circuit_breakers.record_success(provider)
        self._success_stats.observe(provider, success=True)
        self._provider_error_budget_reset_stats.observe(provider, success=True)
        self._provider_weight_stats.observe(provider, success=True)
        self._rate_limit_stats.observe(provider, rate_limited=False)
        self._retry_after_cooldown.clear(provider)
        state_machine.transition(RequestState.RESPONDED)
        latency_ms = (time.perf_counter() - started_at) * 1000.0
        latency_seconds = latency_ms / 1000.0
        scoreboard_latency_ms = (
            (time.perf_counter() - attempt_started_at) * 1000.0
            if attempt_started_at is not None
            else latency_ms
        )
        self._fallback_scoreboard.record_outcome(provider, True, scoreboard_latency_ms)
        self._latency_stats.observe(provider, latency_ms)
        self._latency_slope_stats.observe(provider, latency_ms)
        self._budget_guardrail.record_spend(request.user_id, provider_response.cost_usd)
        self._spend_ledger.record(
            request_id=request.request_id,
            tenant=request.user_id,
            provider=provider,
            model=provider_response.model,
            cost_usd=provider_response.cost_usd,
            input_tokens=provider_response.input_tokens,
            output_tokens=provider_response.output_tokens,
        )
        self._tenant_budget_cascade_stats.record_spend(
            TenantBudgetCascadeStrategy._tenant_key(request),
            provider_response.cost_usd,
        )
        self._family_spend_window.record(provider, provider_response.cost_usd)
        self._provider_hourly_spend_window.record(provider, provider_response.cost_usd)
        self._token_rpm_window.record(provider, prompt_tokens_estimate)
        region_label = (request.region or "global").strip().lower()
        self._region_failover_hysteresis_stats.record_success(region_label)
        total_tokens = provider_response.input_tokens + provider_response.output_tokens
        if total_tokens > 0:
            self._cost_anomaly_stats.observe((provider_response.cost_usd / total_tokens) * 1000.0)
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
            finish_reason=provider_response.finish_reason,
        )
        audit_record = AuditRecord(
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
            api_key_id=request.api_key_id,
            tenant_id=request.tenant_id,
            domain=domain,
        )
        self._audit_log.append(audit_record)
        self._register_for_feedback(audit_record)
        self._logger.info(
            "routing_response",
            request_id=request.request_id,
            model=response.model_used,
            latency_ms=response.latency_ms,
            cost_usd=response.cost_usd,
        )
        return response

    def _register_for_feedback(self, record: AuditRecord) -> None:
        """Make a completed request addressable by ``POST /v1/feedback``.

        A failing feedback store never fails the request: the provider already
        answered, and treating the error as a provider failure would retry (and
        bill) a fallback. Feedback for this request then returns 404.
        """
        try:
            self._feedback_store.register_request(record)
        except (sqlite3.Error, OSError) as error:
            self._logger.warning(
                "feedback_index_failed", request_id=record.request_id, error=str(error)
            )

    def _client_rejection(
        self,
        exception: Exception,
        request: RouterRequest,
        provider: str,
        model_name: str,
        state_machine: RoutingStateMachine,
    ) -> ProviderRejectedRequestError | None:
        """Return the error to raise when a provider rejected the request itself.

        HTTP 400, 413, and 422 describe the request, not provider health. Counting
        them as provider failures would let a few malformed requests from one
        client open every provider's circuit breaker and fail all other traffic,
        and retrying them on fallback providers would only repeat the rejection.
        They are logged, but no circuit-breaker, success-rate, scoreboard, or
        error-rate signal is recorded and the fallback chain stops.

        Args:
            exception: Failure raised by the provider attempt.
            request: Router request.
            provider: Provider that answered.
            model_name: Model that was attempted.
            state_machine: Request lifecycle state machine.

        Returns:
            The rejection to raise, or ``None`` when the failure is a provider fault.
        """
        status_code = exception.status_code if isinstance(exception, ProviderError) else None
        if status_code is None or status_code not in _CLIENT_ERROR_STATUS_CODES:
            return None
        if state_machine.current_state is RequestState.DISPATCHED:
            state_machine.transition(RequestState.FAILED)
        self._logger.warning(
            "provider_request_rejected",
            request_id=request.request_id,
            provider=provider,
            model=model_name,
            status_code=status_code,
            error=str(exception),
        )
        return ProviderRejectedRequestError(
            f"provider {provider} rejected the request (HTTP {status_code}): {exception}",
            status_code=status_code,
        )

    def _order_attempts(self, chosen_model: str, fallback_chain: Sequence[str]) -> list[str]:
        """Keep the strategy primary model, reorder fallbacks by provider health.

        Uses ``ProviderFallbackScoreboard.rank()`` so healthier providers are
        tried earlier in the fallback chain. Unknown providers keep their
        relative input order after known scored providers.
        """
        if not fallback_chain:
            return [chosen_model]
        provider_order: list[str] = []
        models_by_provider: dict[str, list[str]] = {}
        for model_name in fallback_chain:
            provider = self._model_catalog[model_name].provider
            if provider not in models_by_provider:
                provider_order.append(provider)
                models_by_provider[provider] = []
            models_by_provider[provider].append(model_name)
        ranked_providers = self._fallback_scoreboard.rank(provider_order)
        ordered_fallbacks = [
            model_name
            for provider in ranked_providers
            for model_name in models_by_provider[provider]
        ]
        return [chosen_model, *ordered_fallbacks]

    @staticmethod
    def _tenant_inflight_key(request: RouterRequest) -> str:
        """Resolve the tenant/session key used for tenant-scoped InflightStats."""
        metadata = request.metadata
        for key in ("tenant_id", "user_id", "sticky_key"):
            value = metadata.get(key)
            if value is not None and str(value).strip():
                return str(value)
        if request.user_id != "anonymous":
            return request.user_id
        return request.session_id

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

    @staticmethod
    def _extract_retry_after_seconds(exception: Exception) -> float | None:
        """Parse a numeric Retry-After wait from a provider exception when present."""
        candidates: list[object] = [
            getattr(exception, "retry_after", None),
            getattr(exception, "retry_after_seconds", None),
        ]
        headers = getattr(exception, "headers", None)
        if isinstance(headers, Mapping):
            candidates.append(headers.get("Retry-After") or headers.get("retry-after"))
        response = getattr(exception, "response", None)
        response_headers = getattr(response, "headers", None)
        if isinstance(response_headers, Mapping):
            candidates.append(
                response_headers.get("Retry-After") or response_headers.get("retry-after")
            )
        for candidate in candidates:
            if candidate is None:
                continue
            try:
                return max(0.0, float(str(candidate)))
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _is_rate_limit_error(exception: Exception) -> bool:
        """Return whether a provider error carries a 429/rate-limit signal."""
        status_candidates: list[object] = [
            getattr(exception, "status_code", None),
            getattr(exception, "status", None),
            getattr(getattr(exception, "response", None), "status_code", None),
        ]
        if 429 in status_candidates or "429" in {str(status) for status in status_candidates}:
            return True

        message = str(exception).lower()
        return any(
            marker in message
            for marker in (
                "rate limit",
                "rate-limit",
                "rate_limit",
                "too many requests",
                "http 429",
                "status 429",
            )
        )
