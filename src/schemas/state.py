"""State schema for CMN-C1-009 ClassificationAgent.

All fields are primitives or JSON-serialisable strings. No Pydantic, dataclass,
or arbitrary Python objects (msgpack / LangGraph checkpoint safety).
InvocationContext is NOT stored in state.
"""

from typing import Any, Optional

from framework.schemas.agent_state import AgentState


class ClassificationState(AgentState):
    """Flat state for the ClassificationAgent pipeline.

    Field naming convention:
      - Inputs:        input_text, classification_mode, label_list,
                       few_shot_examples, confidence_threshold, max_input_length
      - Normalize:     normalized_text
      - LLM:           llm_raw_output
      - Output:        labels, confidence, review_required, final_output,
                       formatted_output
      - Errors:        error_code, error_message
    """

    # ---------- Input ----------
    input_text: Optional[str]
    """Raw input text to classify."""

    classification_mode: Optional[str]
    """One of: "binary" | "multi-class" | "multi-label"."""

    label_list: Optional[list[Any]]
    """List[str] of valid labels. Populated by caller or config."""

    few_shot_examples: Optional[list[Any]]
    """Optional List[{"input": str, "labels": List[str]}] for few-shot prompting."""

    confidence_threshold: Optional[float]
    """Confidence below this triggers review_required=True. Default: 0.7."""

    max_input_length: Optional[int]
    """Maximum characters allowed in input_text. Defaults to 8000 if absent."""

    # ---------- Normalize ----------
    normalized_text: Optional[str]
    """Output of initialize: stripped, HTML-clean, whitespace-collapsed text."""

    # ---------- LLM ----------
    llm_raw_output: Optional[str]
    """Raw JSON string from the LLM — preserved for debugging."""

    # ---------- Output ----------
    labels: Optional[list[Any]]
    """List[str] of validated label assignments."""

    confidence: Optional[float]
    """LLM-reported confidence score 0.0–1.0."""

    review_required: Optional[bool]
    """True when confidence < confidence_threshold."""

    final_output: Optional[str]
    """JSON string of the classification result delivered to the caller."""

    formatted_output: Optional[str]
    """Post-S-3-gate final payload returned by get_output() as `output`."""

    # ---------- Error propagation ----------
    error_code: Optional[str]
    """Set on fatal error (e.g. "S1_EMPTY_INPUT"). When non-None, downstream
    slots short-circuit and propagate this field."""

    error_message: Optional[str]
    """Human-readable description of the error. Set alongside error_code."""
