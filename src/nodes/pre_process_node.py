"""PreProcessNode for CMN-C1-009 ClassificationAgent.

Validates the caller-supplied label taxonomy and few-shot examples (was
TaxonomyLoad). No external fetch — taxonomy is always caller-provided (S-2).
"""

import logging
from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.services.events import emitter
from shared.services.events.types import EventType
from shared.utils.audit_logger import emit_trace_event

from src.services import classify

logger = logging.getLogger(__name__)


def _halted(state: dict[str, Any]) -> bool:
    return bool(state.get("error_code")) or state.get("status") == AgentStatus.ERROR.value


class PreProcessNode(FunctionNode):
    """Validate label taxonomy + few-shot examples."""

    # S-1 boundary node: the caller-supplied taxonomy enters the graph here, so
    # this node carries the agent-level trust requirement declared in
    # config/agent.yaml (VERIFIED_EXTERNAL). Taxonomy validation is a pure
    # function with no internal resource or partitioned-KB write, so a verified
    # external caller — the level the Marketplace runner supplies — is correct.
    # Declared explicitly — implicit ANONYMOUS inheritance is not acceptable on a
    # FunctionNode (gate-trust-level-check).
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        if _halted(state):
            return {}

        label_list = state.get("label_list")
        mode = state.get("classification_mode", "multi-class")
        few_shot = state.get("few_shot_examples")

        rejection = classify.validate_taxonomy(label_list, mode, few_shot)
        if rejection is not None:
            code, message = rejection
            # S-4: taxonomy rejection is a domain outcome, not a framework
            # lifecycle event. Emit the code only — never the taxonomy values,
            # which are caller content.
            emit_trace_event("taxonomy_rejected", {"error_code": code}, state)
            return {"status": AgentStatus.ERROR.value, "error_code": code, "error_message": message}

        # S-4: counts/mode only — label text is caller content and stays out of
        # the audit payload.
        emit_trace_event(
            "taxonomy_validated",
            {
                "label_count": len(label_list) if isinstance(label_list, list) else 0,
                "classification_mode": mode,
                "few_shot_count": len(few_shot) if isinstance(few_shot, list) else 0,
            },
            state,
        )
        # Progress shown to the caller while the run is in flight. Counts and
        # mode only — label text is caller content.
        emitter().emit_event(
            event_type=EventType.PROGRESS_UPDATE,
            message=f"Taxonomy validated ({len(label_list) if isinstance(label_list, list) else 0} labels, {mode}).",
            metadata={"stage": "pre_process"},
        )
        return {"status": AgentStatus.SUCCESS.value}
