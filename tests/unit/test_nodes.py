"""Unit tests for slot nodes + classification services (CMN-C1-009)."""
import json

import pytest

from framework.schemas.agent_status import AgentStatus
from framework.secrets.base import MissingSecret

from src.nodes.initialize_node import InitializeNode
from src.nodes.main_node import MainNode
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode
from src.services import classify
from src.services.output_gate import scan_output


class _FakeLLM:
    """Stub LLM service returning a canned JSON classification response."""

    def __init__(self, response: str):
        self._response = response

    def generate(self, prompt: str) -> str:
        return self._response


def _req(**over):
    base = {
        "input_text": "This product is excellent, I love it!",
        "label_list": ["positive", "negative"],
        "classification_mode": "multi-class",
    }
    base.update(over)
    return {"user_input": json.dumps(base)}


# ── services ─────────────────────────────────────────────────────────────────


def test_validate_input_ok():
    assert classify.validate_input("hello world", "multi-class", None) is None


def test_validate_input_empty():
    assert classify.validate_input("", "multi-class", None)[0] == "S5_EMPTY_INPUT"


def test_validate_input_too_long():
    assert classify.validate_input("x" * 50, "multi-class", 10)[0] == "S5_INPUT_TOO_LONG"


def test_validate_input_bad_mode():
    assert classify.validate_input("hello", "bogus", None)[0] == "S5_INVALID_MODE"


def test_validate_input_credential():
    assert classify.validate_input("token sk-abcdefghijklmnopqrstuvwxyz12", "multi-class", None)[0] \
        == "S2_CREDENTIAL_DETECTED"


def test_normalize_text_strips_html_and_whitespace():
    assert classify.normalize_text("  <b>hi</b>   there  ") == "hi   there".replace("   ", " ")


def test_validate_taxonomy_ok():
    assert classify.validate_taxonomy(["a", "b"], "binary", None) is None


def test_validate_taxonomy_empty():
    assert classify.validate_taxonomy([], "multi-class", None)[0] == "TAXONOMY_EMPTY"


def test_validate_taxonomy_duplicate():
    assert classify.validate_taxonomy(["a", "a"], "multi-class", None)[0] == "TAXONOMY_DUPLICATE"


def test_validate_taxonomy_binary_arity():
    assert classify.validate_taxonomy(["a", "b", "c"], "binary", None)[0] == "TAXONOMY_BINARY_ARITY"


def test_validate_taxonomy_fewshot_unknown_label():
    fs = [{"input": "x", "labels": ["nope"]}]
    assert classify.validate_taxonomy(["a", "b"], "multi-class", fs)[0] == "FEWSHOT_UNKNOWN_LABEL"


def test_parse_response_valid_json():
    labels, conf, err = classify.parse_response('{"labels": ["positive"], "confidence": 0.9}',
                                                ["positive", "negative"])
    assert labels == ["positive"] and conf == 0.9 and err is None


def test_parse_response_unknown_label():
    labels, conf, err = classify.parse_response('{"labels": ["unknown"], "confidence": 0.9}',
                                                ["positive", "negative"])
    assert labels is None and err is not None


def test_parse_response_unparseable():
    labels, conf, err = classify.parse_response("garbage output", ["positive", "negative"])
    assert labels is None and err is not None


def test_validate_output_arity():
    assert classify.validate_output(["a", "b"], 0.9, "multi-class", ["a", "b"], 0.7)[0] == "OUTPUT_ARITY"


def test_validate_output_not_in_taxonomy():
    assert classify.validate_output(["z"], 0.9, "multi-class", ["a", "b"], 0.7)[0] \
        == "S3_LABEL_NOT_IN_TAXONOMY"


def test_scan_output_blocks_credential():
    assert scan_output("leak sk-abcdefghijklmnopqrstuvwxyz12")[0] == "S3_OUTPUT_CREDENTIAL_DETECTED"


def test_scan_output_clean():
    assert scan_output('{"labels": ["positive"]}') is None


# ── InitializeNode ───────────────────────────────────────────────────────────


class TestInitializeNode:
    def test_valid_seeds_fields(self):
        result = InitializeNode().execute(_req())
        assert result.get("error_code") is None
        assert result["input_text"].startswith("This product")
        assert result["normalized_text"]
        assert result["confidence_threshold"] == 0.7

    def test_empty_input_errors(self):
        result = InitializeNode().execute(_req(input_text=""))
        assert result["status"] == AgentStatus.ERROR
        assert result["error_code"] == "S5_EMPTY_INPUT"

    def test_bad_mode_errors(self):
        result = InitializeNode().execute(_req(classification_mode="bogus"))
        assert result["error_code"] == "S5_INVALID_MODE"


# ── PreProcessNode ───────────────────────────────────────────────────────────


class TestPreProcessNode:
    def test_taxonomy_ok(self):
        state = {"label_list": ["a", "b"], "classification_mode": "binary",
                 "status": AgentStatus.SUCCESS.value}
        result = PreProcessNode().execute(state)
        assert result["status"] == AgentStatus.SUCCESS

    def test_taxonomy_invalid(self):
        state = {"label_list": [], "classification_mode": "multi-class",
                 "status": AgentStatus.SUCCESS.value}
        result = PreProcessNode().execute(state)
        assert result["status"] == AgentStatus.ERROR
        assert result["error_code"] == "TAXONOMY_EMPTY"

    def test_short_circuit_on_error(self):
        assert PreProcessNode().execute({"error_code": "X"}) == {}


# ── MainNode ─────────────────────────────────────────────────────────────────


class TestMainNode:
    def test_classifies(self):
        node = MainNode(llm_service=_FakeLLM('{"labels": ["positive"], "confidence": 0.92}'))
        state = {"normalized_text": "great", "label_list": ["positive", "negative"],
                 "classification_mode": "multi-class", "status": AgentStatus.SUCCESS.value}
        result = node.execute(state)
        assert result["status"] == AgentStatus.SUCCESS
        assert result["labels"] == ["positive"]
        assert result["confidence"] == 0.92

    def test_parse_failure_errors(self):
        node = MainNode(llm_service=_FakeLLM("not json and no labels"))
        state = {"normalized_text": "great", "label_list": ["positive", "negative"],
                 "classification_mode": "multi-class", "status": AgentStatus.SUCCESS.value}
        result = node.execute(state)
        assert result["status"] == AgentStatus.ERROR
        assert result["error_code"] == "LLM_PARSE_FAILED"

    def test_llm_exception_errors(self):
        class _BoomLLM:
            def generate(self, prompt):
                raise RuntimeError("boom")

        node = MainNode(llm_service=_BoomLLM())
        state = {"normalized_text": "x", "label_list": ["a", "b"],
                 "classification_mode": "multi-class", "status": AgentStatus.SUCCESS.value}
        result = node.execute(state)
        assert result["status"] == AgentStatus.ERROR
        assert result["error_code"] == "LLM_INVOCATION_FAILED"

    def test_short_circuit_on_error(self):
        assert MainNode(llm_service=_FakeLLM("x")).execute({"error_code": "X"}) == {}

    def test_missing_secret_degrades_to_deterministic_success(self):
        class _NoSecretLLM:
            def generate(self, prompt):
                raise MissingSecret("AZURE_OPENAI_API_KEY", namespace="agent1000", agent_name="CMN-C1-009")

        node = MainNode(llm_service=_NoSecretLLM())
        state = {"normalized_text": "x", "label_list": ["a", "b"],
                 "classification_mode": "multi-class", "status": AgentStatus.SUCCESS.value}
        result = node.execute(state)
        # No LLM provisioned is a legitimate degrade mode, not an error — must
        # not be conflated with test_llm_exception_errors above (a configured
        # LLM that fails at call time, which stays ERROR).
        assert result["status"] == AgentStatus.SUCCESS
        assert result["labels"] == ["a"]
        assert result["confidence"] == 0.0
        assert result["classification_mode_note"] == "deterministic_fallback_no_llm_provisioned"


# ── PostProcessNode ──────────────────────────────────────────────────────────


class TestPostProcessNode:
    def test_clean_output_passes(self):
        state = {"labels": ["positive"], "confidence": 0.92, "classification_mode": "multi-class",
                 "label_list": ["positive", "negative"], "confidence_threshold": 0.7,
                 "status": AgentStatus.SUCCESS.value}
        result = PostProcessNode().execute(state)
        assert result["status"] == AgentStatus.SUCCESS
        assert result["review_required"] is False
        # final_output is the machine-readable contract a parent Cat 2 agent
        # consumes (docs/02 §8).
        assert json.loads(result["final_output"])["labels"] == ["positive"]

    def test_formatted_output_is_human_readable_not_json(self):
        """`formatted_output` is what AgentBaseGraph.get_output() surfaces as
        `output` — i.e. the chat reply. It must be prose, not the JSON contract:
        returning JSON there is parseable but useless to a reader."""
        state = {"labels": ["positive"], "confidence": 0.92, "classification_mode": "multi-class",
                 "label_list": ["positive", "negative"], "confidence_threshold": 0.7,
                 "status": AgentStatus.SUCCESS.value}
        result = PostProcessNode().execute(state)
        formatted = result["formatted_output"]

        with pytest.raises(json.JSONDecodeError):
            json.loads(formatted)
        assert formatted != result["final_output"]
        assert "positive" in formatted
        assert "92%" in formatted
        assert "Classification result" in formatted

    def test_formatted_output_states_review_when_low_confidence(self):
        state = {"labels": ["positive"], "confidence": 0.4, "classification_mode": "multi-class",
                 "label_list": ["positive", "negative"], "confidence_threshold": 0.7,
                 "status": AgentStatus.SUCCESS.value}
        formatted = PostProcessNode().execute(state)["formatted_output"]
        assert "recommended" in formatted
        assert "40%" in formatted

    def test_low_confidence_sets_review(self):
        state = {"labels": ["positive"], "confidence": 0.4, "classification_mode": "multi-class",
                 "label_list": ["positive", "negative"], "confidence_threshold": 0.7,
                 "status": AgentStatus.SUCCESS.value}
        result = PostProcessNode().execute(state)
        assert result["review_required"] is True

    def test_label_not_in_taxonomy_errors(self):
        state = {"labels": ["zzz"], "confidence": 0.9, "classification_mode": "multi-class",
                 "label_list": ["positive", "negative"], "confidence_threshold": 0.7,
                 "status": AgentStatus.SUCCESS.value}
        result = PostProcessNode().execute(state)
        assert result["status"] == AgentStatus.ERROR
        assert result["error_code"] == "S3_LABEL_NOT_IN_TAXONOMY"

    def test_halt_terminal_error(self):
        result = PostProcessNode().execute({"error_code": "X", "status": AgentStatus.SUCCESS.value})
        assert result["status"] == AgentStatus.ERROR
