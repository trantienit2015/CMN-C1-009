"""PostProcessNode for CMN-C1-009 ClassificationAgent.

Validates the LLM labels against the taxonomy (S-3 allowlist), sets
review_required from the confidence threshold, runs the S-3 output gate on the
serialized result, and emits an audit event (was OutputValidate + the
agent-boundary _security_gate_output).
"""

import json
import logging
from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.services.events import emitter
from shared.services.events.types import EventType
from shared.utils.audit_logger import emit_trace_event

from src.services import classify
from src.services.output_gate import scan_output

logger = logging.getLogger(__name__)


def _halted(state: dict[str, Any]) -> bool:
    return bool(state.get("error_code")) or state.get("status") == AgentStatus.ERROR.value


class PostProcessNode(FunctionNode):
    """Validate labels, set review_required, S-3 output gate, audit."""

    # Downstream node: the S-1 boundary is enforced at PreProcessNode, and this
    # node only reads state the graph already admitted. Declared explicitly —
    # implicit ANONYMOUS inheritance is not acceptable (gate-trust-level-check).
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        if _halted(state):
            emit_trace_event("classification_error", {"error_code": state.get("error_code")}, state)
            return {"status": AgentStatus.ERROR.value}

        labels = state.get("labels")
        confidence = state.get("confidence", 0.0)
        mode = state.get("classification_mode", "multi-class")
        label_list = state.get("label_list", []) or []
        threshold = float(state.get("confidence_threshold") or classify.DEFAULT_CONFIDENCE_THRESHOLD)

        rejection = classify.validate_output(labels, confidence, mode, label_list, threshold)
        if rejection is not None:
            code, message = rejection
            logger.warning("Output validation failed (%s)", code)
            emit_trace_event("classification_rejected", {"error_code": code}, state)
            return {"status": AgentStatus.ERROR.value, "error_code": code, "error_message": message}

        review_required = float(confidence) < threshold

        result = {
            "labels": labels,
            "mode": mode,
            "confidence": confidence,
            "review_required": review_required,
        }
        final_output = json.dumps(result)
        # `final_output` stays the machine-readable JSON contract a parent Cat 2
        # agent consumes (docs/02 §8). `formatted_output` is what a human sees:
        # AgentBaseGraph.get_output() surfaces it as `output`, so it is the chat
        # reply. Returning raw JSON there is readable by a parser, not a person.
        formatted_output = classify.format_result(
            labels if isinstance(labels, list) else [],
            confidence,
            mode,
            review_required,
            threshold,
        )

        # S-3 output gate on BOTH payloads that leave the agent — the JSON
        # contract and the human-facing text. Scanning only the JSON would let
        # the reply surfaced by get_output() bypass the gate.
        violation = scan_output(final_output) or scan_output(formatted_output)
        if violation is not None:
            code, message = violation
            logger.warning("S-3 gate blocked classification output (%s)", code)
            emit_trace_event("classification_blocked", {"error_code": code}, state)
            return {
                "status": AgentStatus.ERROR.value,
                "error_code": code,
                "error_message": message,
                "final_output": None,
                "formatted_output": None,
            }

        emitter().emit_event(
            event_type=EventType.PROGRESS_UPDATE,
            message="Validating labels against the taxonomy.",
            metadata={"stage": "post_process"},
        )

        emit_trace_event(
            "classification_result",
            {
                "label_count": len(labels) if isinstance(labels, list) else 0,
                "review_required": review_required,
            },
            state,
        )
        return {
            "labels": labels,
            "confidence": confidence,
            "review_required": review_required,
            "final_output": final_output,
            "formatted_output": formatted_output,
            "status": AgentStatus.SUCCESS.value,
        }
