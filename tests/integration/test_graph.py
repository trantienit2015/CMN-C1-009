"""Integration tests — full compile() + invoke() over the 5-node backbone (CMN-C1-009)."""
import json

import pytest

from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel

from src.graph.graph import ClassificationAgent
from src.nodes.main_node import MainNode


class _FakeLLM:
    def __init__(self, response: str):
        self._response = response

    def generate(self, prompt: str) -> str:
        return self._response


class _TestableClassificationAgent(ClassificationAgent):
    """Agent variant wiring a deterministic fake LLM into the main slot."""

    def __init__(self, llm_response: str):
        super().__init__()
        self._llm_response = llm_response

    def register_nodes(self) -> None:
        super().register_nodes()
        self._nodes["main"] = MainNode(llm_service=_FakeLLM(self._llm_response))


def _payload(**over):
    base = {
        "input_text": "This product is excellent!",
        "label_list": ["positive", "negative"],
        "classification_mode": "multi-class",
    }
    base.update(over)
    return json.dumps(base)


def _ctx(trust=TrustLevel.INTERNAL):
    return InvocationContext(caller_trust_level=trust)


def test_full_pipeline_success():
    agent = _TestableClassificationAgent('{"labels": ["positive"], "confidence": 0.92}')
    agent.compile()
    result = agent.invoke(_payload(), ctx=_ctx())
    assert result["status"] == AgentStatus.SUCCESS.value
    assert result["output"]
    # get_output() surfaces formatted_output as `output` — the human-facing
    # reply, so assert prose here. The JSON contract lives in final_output and
    # is covered by the PostProcessNode unit tests.
    assert "positive" in result["output"]
    assert "Classification result" in result["output"]
    with pytest.raises(json.JSONDecodeError):
        json.loads(result["output"])


def test_backbone_order():
    agent = _TestableClassificationAgent('{"labels": ["positive"], "confidence": 0.92}')
    agent.compile()
    result = agent.invoke(_payload(), ctx=_ctx())
    history = result["node_history"]
    for node in ("InitializeNode", "PreProcessNode", "MainNode", "PostProcessNode", "FinalizeNode"):
        assert node in history


def test_invalid_input_short_circuits():
    agent = _TestableClassificationAgent('{"labels": ["positive"], "confidence": 0.92}')
    agent.compile()
    result = agent.invoke(_payload(input_text=""), ctx=_ctx())
    assert result["status"] == AgentStatus.ERROR.value
    assert result.get("error_code") == "S5_EMPTY_INPUT"


def test_bad_taxonomy_short_circuits():
    agent = _TestableClassificationAgent('{"labels": ["positive"], "confidence": 0.92}')
    agent.compile()
    result = agent.invoke(_payload(label_list=[]), ctx=_ctx())
    assert result["status"] == AgentStatus.ERROR.value
    assert result.get("error_code") == "TAXONOMY_EMPTY"


def test_low_confidence_sets_review_required():
    agent = _TestableClassificationAgent('{"labels": ["positive"], "confidence": 0.3}')
    agent.compile()
    result = agent.invoke(_payload(), ctx=_ctx())
    assert result["status"] == AgentStatus.SUCCESS.value
    # The reply must actually tell the reader review is needed, not just carry
    # a review_required flag a parser would have to interpret.
    assert "recommended" in result["output"]


def test_s1_trust_gate_blocks_low_trust():
    agent = _TestableClassificationAgent('{"labels": ["positive"], "confidence": 0.92}')
    agent.compile()
    result = agent.invoke(_payload(), ctx=_ctx(trust=TrustLevel.ANONYMOUS))
    assert result["status"] == AgentStatus.ERROR.value
