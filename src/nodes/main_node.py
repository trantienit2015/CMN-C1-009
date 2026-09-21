"""MainNode for CMN-C1-009 ClassificationAgent.

Builds a structured prompt from normalized_text + label_list + few_shot_examples
and calls the LLM, then parses the response into labels + confidence (was
LLMClassify). Routed slot — sets the status route() reads.
"""

import logging
from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel
from framework.secrets.base import MissingSecret
from shared.services.events import emitter
from shared.services.events.types import EventType
from shared.utils.audit_logger import emit_trace_event

from src.services import classify
from src.services.llm_service import ClassificationLLMService

logger = logging.getLogger(__name__)


def _halted(state: dict[str, Any]) -> bool:
    return bool(state.get("error_code")) or state.get("status") == AgentStatus.ERROR.value


class MainNode(FunctionNode):
    """Call the LLM to classify normalized_text against the label_list."""

    # Downstream node: the S-1 boundary is enforced at PreProcessNode, and this
    # node only reads state the graph already admitted. Declared explicitly —
    # implicit ANONYMOUS inheritance is not acceptable (gate-trust-level-check).
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def __init__(self, llm_service: ClassificationLLMService | None = None):
        # An injected service (tests / explicit wiring) is used as-is. Otherwise
        # the LLM client is built lazily inside execute() from the credential
        # resolved via ctx.secrets — node instances are shared across every
        # invocation, so the authenticated client must NOT be cached on self.
        self._injected = llm_service

    def _llm_service(self, state: dict[str, Any]) -> ClassificationLLMService:
        if self._injected is not None:
            return self._injected
        ctx = InvocationContext.from_state(state)
        # Azure OpenAI: api_key is a provisioned secret; endpoint + deployment are
        # deployment-time env vars supplied through the same secret provider
        # (registered as Marketplace env_vars). All three are required by
        # AzureOpenAIClient — missing any one raises a clear ValueError.
        config = {
            "api_key": ctx.secrets.require("AZURE_OPENAI_API_KEY"),
            "azure_endpoint": ctx.secrets.require("AZURE_OPENAI_ENDPOINT"),
            "azure_deployment": ctx.secrets.require("AZURE_OPENAI_DEPLOYMENT"),
        }
        return ClassificationLLMService(config=config)

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        if _halted(state):
            return {}

        normalized_text = state.get("normalized_text") or ""
        label_list = state.get("label_list", []) or []
        mode = state.get("classification_mode", "multi-class")
        few_shot = state.get("few_shot_examples")

        prompt = classify.build_prompt(normalized_text, label_list, mode, few_shot)

        emitter().emit_event(
            event_type=EventType.PROGRESS_UPDATE,
            message="Classifying with Azure OpenAI.",
            metadata={"stage": "main"},
        )

        # A missing provisioned secret (LLM API key not configured at all,
        # e.g. STG provisional CI) is a legitimate degrade mode, not an error —
        # distinct from a configured LLM that fails at call time (still ERROR
        # below). Fall back to a deterministic, clearly-marked classification
        # instead of failing the whole invocation.
        try:
            raw_output = self._llm_service(state).generate(prompt)
        except MissingSecret:
            emit_trace_event(
                "classification_deterministic_fallback",
                {"reason": "no_llm_provisioned"},
                state,
            )
            fallback_labels, fallback_confidence = classify.deterministic_fallback(label_list, mode)
            return {
                "labels": fallback_labels,
                "confidence": fallback_confidence,
                "status": AgentStatus.SUCCESS.value,
                "classification_mode_note": "deterministic_fallback_no_llm_provisioned",
            }
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "LLMClassify: LLM call failed — session=%s error=%s",
                state.get("session_id"),
                exc,
                exc_info=True,
            )
            # S-4: the external LLM call is this node's side effect, so its
            # failure is a domain event. Emit the error code only — the
            # exception text can carry provider-side content.
            emit_trace_event("llm_invocation_failed", {"error_code": "LLM_INVOCATION_FAILED"}, state)
            return {
                "status": AgentStatus.ERROR.value,
                "error_code": "LLM_INVOCATION_FAILED",
                "error_message": f"LLM call failed: {exc}",
            }

        parsed_labels, parsed_confidence, err = classify.parse_response(raw_output, label_list)
        if err or parsed_labels is None or parsed_confidence is None:
            emit_trace_event("llm_parse_failed", {"error_code": "LLM_PARSE_FAILED"}, state)
            return {
                "llm_raw_output": raw_output,
                "status": AgentStatus.ERROR.value,
                "error_code": "LLM_PARSE_FAILED",
                "error_message": err or "LLM output could not be parsed into valid labels",
            }
        labels: list[Any] = parsed_labels
        confidence: float = parsed_confidence

        # S-4: label count + confidence only — the classified text and the
        # assigned label values are caller/model content, not audit payload.
        emit_trace_event(
            "classification_completed",
            {"label_count": len(labels) if isinstance(labels, list) else 0, "confidence": confidence},
            state,
        )
        return {
            "llm_raw_output": raw_output,
            "labels": labels,
            "confidence": confidence,
            "status": AgentStatus.SUCCESS.value,
        }
