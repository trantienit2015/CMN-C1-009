# 02_design.md — CMN-C1-009 ClassificationAgent Design Specification

## 1. Overview

`ClassificationAgent` is a generic, industry-agnostic Cat 1 classification primitive built on the AgentCore L1 framework. It assigns one or more labels from a caller-provided taxonomy to a given input text using an LLM, with full security guardrails and config-driven taxonomy.

**Template ID:** CMN-C1-009  
**Category:** C1 (generic component)  
**Industry:** CMN  
**L1 Base:** `AgentBaseGraph` (direct framework inheritance — per 2026-05-18 policy). The graph class IS the agent; no separate agent class, no `_invoke_impl`, no `.run()`.
**Level 2 type:** N/A — per 2026-05-18 policy, Level 2 inheritance is retired for agent-templates; this template inherits directly from the Level 1 (L1) framework. Level 2 base agents (RAGAgent, ChatAgent, etc.) are no longer used as inheritance targets.  
**Reuse target:** Cat 2 domain templates that need text classification (e.g. RET-C2-*, FIN-C2-*)

---

## 2. Pipeline

Fixed AgentCore 5-node backbone (the original 4-node linear pipeline collapsed into
the fillable slots):

```
START → initialize → pre_process → main → {route} → post_process → finalize → END
```

| Slot | Maps from | Role |
|---|---|---|
| `initialize` | InputNormalize | Parse request, S-5/S-2 input validation, normalize text |
| `pre_process` | TaxonomyLoad | Validate caller-supplied label_list + few_shot_examples |
| `main` | LLMClassify | Build prompt → LLM classify → parse labels + confidence (routed slot) |
| `post_process` | OutputValidate | Label/arity validation, `review_required`, S-3 output gate, audit |

Downstream slots short-circuit (`return {}`) once `error_code` / `status=ERROR` is set
upstream. `finalize` is framework-owned.

---

## 3. State Schema (`ClassificationState`)

All fields are primitives / JSON-serializable. No Pydantic. No InvocationContext in state.

| Field | Type | Description |
|---|---|---|
| `session_id` | `str` | Caller-supplied session identifier |
| `input_text` | `str` | Raw input text to classify |
| `classification_mode` | `str` | `"binary"` \| `"multi-class"` \| `"multi-label"` |
| `label_list` | `list[str]` | Labels to assign from (set by caller or config) |
| `few_shot_examples` | `Optional[list]` | `[{"input": str, "labels": list[str]}]` |
| `confidence_threshold` | `float` | Below this → `review_required=True`. Default: 0.7 |
| `normalized_text` | `Optional[str]` | Output of InputNormalize |
| `llm_raw_output` | `Optional[str]` | Raw LLM JSON string (debugging) |
| `labels` | `Optional[list[str]]` | Final validated label assignments |
| `confidence` | `Optional[float]` | LLM-reported confidence 0.0–1.0 |
| `review_required` | `Optional[bool]` | `True` when `confidence < confidence_threshold` |
| `error_message` | `Optional[str]` | Set on any validation/security failure |

---

## 4. Node Contracts

### 4.1 InputNormalize

**Input:** `session_id`, `input_text`, `classification_mode`, `confidence_threshold`  
**Output:** `normalized_text` (or `error_message`)

- S-1: enforce `INTERNAL` trust from `config["configurable"]["invocation_context"]`
- Validate `input_text` non-empty, within `max_input_length` (8,000 chars)
- Validate `classification_mode` ∈ `{binary, multi-class, multi-label}`
- Normalize: strip HTML, collapse whitespace, strip leading/trailing whitespace
- Store as `normalized_text`

### 4.2 TaxonomyLoad

**Input:** `label_list`, `classification_mode`, `few_shot_examples`  
**Output:** validated label_list in state (or `error_message`)

- Validate `label_list` non-empty, no duplicates, all non-empty strings
- For `binary` mode: exactly 2 labels
- Validate `few_shot_examples` if present: each `{input: str, labels: list[str]}`, all labels in `label_list`
- No external fetch — taxonomy must be caller-provided (S-2 constraint)

### 4.3 LLMClassify

**Input:** `normalized_text`, `label_list`, `few_shot_examples`, `classification_mode`  
**Output:** `llm_raw_output`, `labels`, `confidence` (or `error_message`)

- Build mode-specific prompt:
  - `binary`: "Answer YES or NO only."
  - `multi-class`: "Select exactly one label from the list."
  - `multi-label`: "Select all applicable labels from the list."
- Apply `sanitize_query()` injection guard before LLM call
- Target LLM JSON: `{"labels": [...], "confidence": 0.0–1.0}`
- Defensive parsing: strip markdown fences → `json.loads` → validate → fallback label search
- Reject unknown labels (not in `label_list`)

### 4.4 OutputValidate

**Input:** `labels`, `confidence`, `classification_mode`, `label_list`, `confidence_threshold`  
**Output:** `review_required`, S-3 gate check (or `error_message`)

- Validate label count: binary/multi-class → exactly 1; multi-label → ≥ 1
- Allowlist check: all labels must be in `label_list` (S-3 taxonomy gate)
- Set `review_required = confidence < confidence_threshold`
- Apply `apply_s3_gate()` on JSON-serialized output — `error_message` on BLOCKER

---

## 5. Taxonomy YAML Schema (`config/agent.yaml`)

```yaml
template_id: "CMN-C1-009"
name: "ClassificationAgent"
category: "C1"
industry: "CMN"
level2_base: "L1"
version: "0.1.0"
required_trust_level: "INTERNAL"
classification_mode: "multi-class"
confidence_threshold: 0.7
max_input_length: 8000
# label_list: []       # set per deployment
# few_shot_examples: []  # optional
```

Callers must supply `label_list` at invocation time (or pre-populate state).

---

## 6. Confidence Threshold & `review_required` Flag

- Default threshold: 0.7
- If LLM-reported `confidence < threshold` → `review_required = True`
- Callers can use `review_required` to route to human review queues
- Threshold is configurable per deployment via `config/agent.yaml`

---

## 7. Security Placement

| Security Layer | Node | Implementation |
|---|---|---|
| S-1 (Trust) | InputNormalize + ClassificationAgent.__pre_invoke__ | `INTERNAL` required |
| S-2 (Input gate) | LLMClassify | `sanitize_query(strip_html=True)` before LLM |
| S-3 (Output gate) | OutputValidate | `apply_s3_gate()` on serialized output |
| S-4 (Trace) | ClassificationAgent | `_emit_trace_event()` always fires |
| S-5 (Fail-fast) | InputNormalize | Truncation/empty input → immediate error_message |

---

## 8. Relationship with CMN-C1-048

- CMN-C1-009: pure label assignment → outputs `{labels, confidence, review_required}`
- CMN-C1-048: routing/orchestration agent → uses classification results to branch workflows
- Use CMN-C1-009 when you need classification output as data; use CMN-C1-048 when you need routing decisions

---

## 9. Error Strategy

Early-exit on `error_message`. Any node that sets `error_message` halts the pipeline. Downstream nodes check and return immediately if `error_message` is set.
