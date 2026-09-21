# AgentCore Platform v1.0
"""AGENTIC STAR Marketplace entrypoint — one-shot Pod process for CMN-C1-009.

Referenced by the Dockerfile's Marketplace CMD. run_agent_marketplace() owns the
whole lifecycle: it constructs the graph, compiles it (attaching a checkpointer
when config enables memory/HITL), builds the SecretProvider, provisions secrets,
then runs one Marketplace execution and exits.

`config` MUST be passed: run_agent_marketplace() falls back to
`agent_cls(config={})`, so omitting it silently drops config/config.yaml and
every self.config.get(...) reads None. load_agent_config() reads that file.

`agent_name` / `namespace` mirror config/agent.yaml (`id` / `namespace`) so the
Pod is identifiable in the Marketplace runner. Note they do NOT scope secret
lookup: the Marketplace EnvProvider reads os.environ unscoped (one Pod runs one
agent, so the process is the isolation boundary) — env var names are the
platform's own contract.
"""

from pathlib import Path

from framework.utils.config_loader import load_agent_config
from shared.bootstrap.marketplace_app import run_agent_marketplace

from src.graph.graph import ClassificationAgent

# Runtime overrides that should not live in config/config.yaml go here.
extend_config: dict = {}

if __name__ == "__main__":
    run_agent_marketplace(
        ClassificationAgent,
        agent_name="CMN-C1-009",
        namespace="cmn",
        config={**load_agent_config(Path(__file__).resolve().parent), **extend_config},
    )
