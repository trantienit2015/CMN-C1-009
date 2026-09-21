"""InitializeNode for CMN-C1-009 ClassificationAgent.

Runs first. Parses the request payload (JSON string in user_input) into the
domain state, then runs S-5/S-2 input validation and text normalization (was
InputNormalize). Fail fast: sets error_code/error_message on rejection.
"""

from typing import Any
import json

from framework.nodes.defaults.initialize_node import InitializeNode as DefaultInitializeNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.services import classify


class InitializeNode(DefaultInitializeNode):
    """Parse + S-5/S-2 validate + normalize the classification request."""

    # S-1: generic text classification with a caller-supplied taxonomy — no
    # internal resource or partitioned KB write. Matches the agent-level trust
    # in config/agent.yaml (VERIFIED_EXTERNAL): the Marketplace runner supplies
    # a verified external caller, which is the correct level for a public,
    # use-case-agnostic classification capability.
    required_trust_level = TrustLevel.VERIFIED_EXTERNAL

    def __init__(self, default_label_list: Any = None, default_mode: Any = None, default_threshold: Any = None):
        # Deployment defaults from config/config.yaml. Immutable per-instance
        # config, not per-invocation state — a caller-supplied
        # value in the payload always takes precedence over these.
        super().__init__()
        self._default_label_list = list(default_label_list) if default_label_list else None
        self._default_mode = default_mode
        self._default_threshold = default_threshold

    def on_initialize(self, state: dict[str, Any]) -> dict[str, Any]:
        fields = self._extract_fields(state)
        input_text = fields.get("input_text")
        # Caller payload wins; deployment config fills the gap (docs/02 §5).
        mode = fields.get("classification_mode") or self._default_mode or "multi-class"
        max_len = fields.get("max_input_length")

        rejection = classify.validate_input(input_text, mode, max_len)
        if rejection is not None:
            code, message = rejection
            return {"status": AgentStatus.ERROR.value, "error_code": code, "error_message": message}

        normalized = classify.normalize_text(input_text if isinstance(input_text, str) else "")

        threshold = fields.get("confidence_threshold")
        if threshold is None:
            threshold = self._default_threshold
        if threshold is None:
            threshold = classify.DEFAULT_CONFIDENCE_THRESHOLD

        label_list = fields.get("label_list") or self._default_label_list

        return {
            "input_text": input_text,
            "classification_mode": mode,
            "label_list": label_list,
            "few_shot_examples": fields.get("few_shot_examples"),
            "confidence_threshold": float(threshold),
            "max_input_length": max_len,
            "normalized_text": normalized,
        }

    @staticmethod
    def _extract_fields(state: dict[str, Any]) -> dict[str, Any]:
        payload = {}
        raw = state.get("user_input", "")
        if isinstance(raw, dict):
            payload = raw
        elif isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    payload = parsed
                else:
                    # Valid JSON but not an object (a bare string or number) —
                    # treat it as the text to classify, not as a payload.
                    payload = {"input_text": raw}
            except (ValueError, TypeError):
                # Plain prose, which is what a person types in chat. A parent
                # agent sends the JSON payload; a human sends a sentence, and
                # rejecting that as S5_EMPTY_INPUT makes the agent look broken.
                # The taxonomy then has to come from config (see _default_labels).
                payload = {"input_text": raw}

        def pick(key: Any, default: Any = None) -> Any:
            val = state.get(key)
            if val is not None:
                return val
            return payload.get(key, default)

        return {
            "input_text": pick("input_text"),
            "classification_mode": pick("classification_mode"),
            "label_list": pick("label_list"),
            "few_shot_examples": pick("few_shot_examples"),
            "confidence_threshold": pick("confidence_threshold"),
            "max_input_length": pick("max_input_length"),
        }
