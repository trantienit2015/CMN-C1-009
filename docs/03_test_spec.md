# 03_test_spec.md — CMN-C1-009 ClassificationAgent Test Specification

## 1. Scope

Tests for `ClassificationAgent` covering all 4 nodes plus graph, agent, and security boundaries.

---

## 2. Integration Test Cases (`tests/test_classification.py`)

| TC-ID | Target | Input | Expected |
|---|---|---|---|
| TC-01 | InputNormalize | `input_text=""` | `error_message` set |
| TC-02 | InputNormalize | `input_text` > `max_input_length` (8001 chars) | `error_message` set |
| TC-03 | InputNormalize | `classification_mode="unknown_mode"` | `error_message` set |
| TC-04 | TaxonomyLoad | `label_list=[]` | `error_message` set |
| TC-05 | TaxonomyLoad | `binary` mode with 3 labels | `error_message` set |
| TC-06 | TaxonomyLoad | few_shot example references unknown label | `error_message` set |
| TC-07 | LLMClassify | LLM returns unknown label in JSON | `error_message` set |
| TC-08 | OutputValidate | `confidence=0.5`, `threshold=0.7` | `review_required=True` |
| TC-09 | LLMClassify+OutputValidate | `multi-label` mode, LLM returns 2 valid labels | `labels` has 2 entries |
| TC-10 | OutputValidate | S-3 gate: credential pattern in output | `error_message` with BLOCKER |

---

## 3. Proof-of-Boundary Tests (`tests/proof_of_boundary/test_pb_classification.py`)

| PB-ID | Boundary | Test | Expected |
|---|---|---|---|
| PB-1 | BaseNode → EventEmitter | `_emit_trace_event` fires on invoke | No silent failures; no exception |
| PB-2 | State serialization | Post-invoke state is JSON-serializable | `json.dumps(state)` succeeds |
| PB-3 | L1 → External service | Mock `LLMClient.generate` | No real API calls in CI |
| PB-4 | Import isolation | AST scan of `src/` | Zero `agenticstar` imports |
| PB-5 | Checkpoint safety | No credential patterns in `labels`/`confidence` fields | Checkpoint inspection pass |
| PB-6 | Invoke execution order | Sequence: pre_invoke → _invoke_impl → trace | Order verified via mock/spy |

---

## 4. Framework Compliance Tests (TC-01–TC-08)

| TC-ID | Test | Expected |
|---|---|---|
| FC-01 | State contract | `ClassificationState` is flat TypedDict, no Pydantic/dataclass |
| FC-02 | SecurityViolationError | Raised when trust level is wrong |
| FC-03 | No credentials in state | `label_list`, `labels`, `llm_raw_output` contain no JWT/API keys |
| FC-04 | InvocationContext via configurable | `config["configurable"]["invocation_context"]` only |
| FC-05 | Trace fires | `_emit_trace_event` called on every invoke |
| FC-06 | `_security_gate_input` non-bypassable | `sanitize_query` always runs in LLMClassify |
| FC-07 | `_security_gate_output` non-bypassable | `apply_s3_gate` always runs in OutputValidate |
| FC-08 | Trust level enforced | Insufficient trust → `SecurityViolationError` |
