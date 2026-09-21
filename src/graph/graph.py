"""Graph definition for CMN-C1-009 ClassificationAgent.

Cat 1 — single capability (text classification against a caller-provided
taxonomy), fixed 5-node backbone. The graph class IS the agent (inherits
AgentBaseGraph L1 direct); no separate agent class, no _invoke_impl, no .run().

    START → initialize → pre_process → main → {route} → post_process → finalize → END

Slot mapping (original 4-node linear pipeline collapsed into the backbone):
    initialize   = parse request + S-5/S-2 input validation + normalize (InputNormalize)
    pre_process  = taxonomy + few-shot validation (TaxonomyLoad)
    main         = build prompt + LLM call + parse response (LLMClassify)
    post_process = label/arity validation + review_required + S-3 output gate (OutputValidate)
"""

from typing import Any
from framework.graph.agent_base_graph import AgentBaseGraph

from src.nodes.initialize_node import InitializeNode
from src.nodes.main_node import MainNode
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode
from src.schemas.state import ClassificationState


class ClassificationAgent(AgentBaseGraph):
    """CMN-C1-009 — generic text classification agent (Cat 1, CMN)."""

    @property
    def name(self) -> str:
        return "ClassificationAgent"

    @property
    def state_schema(self) -> type:
        return ClassificationState

    def register_nodes(self) -> None:
        super().register_nodes()
        # Deployment defaults from config/config.yaml (docs/02 §5: taxonomy is
        # "set per deployment"). Passed as immutable constructor config, never
        # cached mutable state on the node. A caller-supplied
        # label_list in the payload still wins — see InitializeNode.
        self._nodes["initialize"] = InitializeNode(
            default_label_list=self.config.get("label_list"),
            default_mode=self.config.get("classification_mode"),
            default_threshold=self.config.get("confidence_threshold"),
        )
        self._nodes["pre_process"] = PreProcessNode()
        self._nodes["main"] = MainNode()
        self._nodes["post_process"] = PostProcessNode()

    def _enrich_output(self, output: dict[str, Any], result: dict[str, Any]) -> None:
        for key in ("final_output", "labels", "confidence", "review_required", "error_code", "error_message"):
            value = result.get(key)
            if value is not None:
                output[key] = value
