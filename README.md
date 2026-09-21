# CMN-C1-009 — ClassificationAgent

> **Category**: Cat 1 (a single, reusable technical capability, independent of any particular use case)
> **Industry**: CMN (cross-industry / generic)

## Overview

This template classifies a piece of text against a taxonomy you define. You give it some
text — a support message, a review, a ticket, a document excerpt — together with the set of
labels you care about, and it returns the label or labels that apply, a confidence score, and
a flag telling you whether the result is confident enough to act on automatically or should be
looked at by a person.

Three modes are supported and are chosen through configuration rather than code: binary
(does this belong to the category or not), multi-class (exactly one label from the set), and
multi-label (any number of labels). The taxonomy itself is configuration too, so adapting the
agent to a new domain usually means editing YAML and prompt text rather than writing Python.

It is deliberately narrow: it assigns labels and stops there. It does not route, notify, or act
on the result. That makes it a building block — use it as the classification step inside a
larger workflow of your own, and keep the decision about what to do with a label in your code,
where your business rules live.

This is an agent template built with the **AGENTIC STAR** development platform and the
**AgentCore Framework**. It is intended to be taken as a starting point: fork it, adapt it to
your own data and policies, and run it inside your own AGENTIC STAR deployment.

## Requirements

**This template does not run standalone.** It requires:

| Requirement | Notes |
|---|---|
| **AGENTIC STAR platform** | The agent connects to the platform at start-up. Without it, start-up fails immediately (see *Behaviour without the platform* below). Deployment guides and API documentation: [AGENTIC STAR Developers](https://developers.fd.agenticstar.tm.softbank.jp/) |
| **AgentCore Framework** (`agenticstar-agentcore`) | Installed from PyPI as a dependency. |
| Python | >=3.11 |

```bash
pip install -e .
```

### Behaviour without the platform

The framework is designed to run **only** on AGENTIC STAR. There is no fallback or degraded
mode. If the platform is unreachable or the SDK version does not match, the agent raises
`PlatformRequired` during graph compile / start-up preflight rather than starting in a partially
working state. This is intentional — a half-running agent is worse than one that refuses to start.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Tests run without a platform connection. Running the agent itself does not.

## Project Structure

```
src/          agent implementation (nodes, services, schemas)
tests/        unit, integration and boundary tests
config/       agent configuration
docs/         design and operational documentation
```

See `docs/` for the proposal, design, test specification, release notes and operation guide.

## Customising

1. Adjust `config/` for your own environment and policies.
2. Replace the knowledge sources and sample data with your own.
3. Review the node implementations under `src/nodes/` for domain-specific logic.
4. Re-run the test suite.

## License

MIT — see [LICENSE](LICENSE).

## Status of this repository

This template is published **as is**, by its individual author, under the MIT license. It carries
**no warranty and no support commitment**, and no organisation stands behind its behaviour or
fitness for any purpose. Issues and pull requests may or may not receive a response; that is at
the sole discretion of the repository owner.

---

