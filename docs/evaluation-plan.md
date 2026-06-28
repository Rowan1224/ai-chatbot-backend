# Evaluation Plan

This document defines the evaluation strategy for two concerns that the
current test suite does not address:

1. **Agent reliability** — how do we measure that the conversational agent
   reliably and accurately completes the task?
2. **Duplicate detection quality** — how do we know the three-stage
   pipeline has acceptable precision and recall?

Both sections follow the same structure: what to measure, how to measure
it, and what tooling to use.

---

## 1. Agent Reliability Evaluation

### What the current tests cover

| Test layer | What it proves |
|---|---|
| Unit (`test_workflow.py`) | Routing logic only — no LLM calls |
| Integration (`test_workflow_integration.py`) | Workflow plumbing with mock DB |
| E2E (`test_full_conversation.py`) | HTTP API end-to-end with a **mock LLM** — fixed scripted responses |

**The gap:** None of these tests the actual LLM behaviour. They verify
that the graph wiring is correct but say nothing about whether the real
model reliably collects fields, refuses injections, or produces well-formed
structured output.

### Dimensions to evaluate

| Dimension | Definition | Target |
|---|---|---|
| **Task completion rate** | Fraction of valid conversations that reach `is_complete=True` | ≥ 99 % |
| **Extraction accuracy** | Fraction of fields extracted correctly vs ground-truth labels | ≥ 95 % per field |
| **Turn efficiency** | Mean turns-to-completion on valid inputs (prompt says 5 fields → expect 5–8 turns) | ≤ 8 turns |
| **Guardrail robustness** | Fraction of injection attempts blocked by the regex layer or the LLM provider | 100 % combined (see note) |
| **Graceful degradation** | Fraction of ambiguous inputs that are clarified rather than silently misfiled | ≥ 90 % |

> **Guardrail robustness note.** The regex layer must catch 100% of the common jailbreak patterns it is designed to detect (these are tested in unit tests). Novel or obfuscated injections that bypass the regex layer are expected to be caught by the LLM provider's own content policy (OpenAI, Azure OpenAI, and Anthropic all enforce this server-side). The `chat_node` handles provider rejections via `BadRequestError`. The combined coverage target is 100%; rigorous regex enumeration of every possible injection variant is not required.

### Golden dataset

A **golden dataset** is a set of labelled multi-turn conversations with
known correct outcomes. It is the foundation of all LLM-level evaluation.

**Structure of one golden record:**

```json
{
  "id": "golden-001",
  "description": "Happy path — infrastructure provisioning to production",
  "turns": [
    {"role": "user", "content": "I need to provision new servers."},
    {"role": "user", "content": "infrastructure-provisioning"},
    {"role": "user", "content": "production"},
    {"role": "user", "content": "Payment service scaling for Q4 peak."},
    {"role": "user", "content": "Jane Smith, EMP4421"}
  ],
  "expected": {
    "is_complete": true,
    "collected_data": {
      "request_type": "infrastructure-provisioning",
      "target_environment": "production",
      "business_justification": "Payment service scaling for Q4 peak.",
      "name": "Jane Smith",
      "employee_id": "EMP4421"
    }
  }
}
```

**Dataset composition (recommended minimum 50 records):**

| Category | Count | Purpose |
|---|---|---|
| Happy path (one field per turn) | 15 | Baseline completion rate |
| Happy path (multiple fields in one turn) | 10 | Tests extraction robustness |
| Ambiguous request type | 5 | Tests normalisation to enum |
| Off-topic first message | 5 | Tests scope enforcement |
| Prompt injection attempts | 10 | Tests guardrail + prompt hardening |
| Mid-conversation topic change | 5 | Tests conversation recovery |

### Implementation approach

**Option A — pytest with real LLM (gated behind `--eval` flag)**

```python
# tests/eval/test_agent_golden.py
import pytest
import json
from pathlib import Path

pytestmark = pytest.mark.eval  # skipped in normal CI

GOLDEN = json.loads(
    Path("tests/eval/golden_conversations.json").read_text()
)

@pytest.mark.parametrize("record", GOLDEN)
async def test_golden_conversation(record, api_client):
    sid = f"eval-{record['id']}"
    final = None
    for turn in record["turns"]:
        final = await api_client.post(
            "/chat",
            json={"session_id": sid, "message": turn["content"]},
        )
    expected = record["expected"]
    assert final["is_complete"] == expected["is_complete"]
    for field, value in expected["collected_data"].items():
        assert final["collected_data"].get(field) == value, (
            f"Field '{field}' mismatch in {record['id']}"
        )
```

**Option B — LangSmith evaluation datasets**

LangSmith tracing is already wired into the project (`langsmith_tracing`
in `settings.py`). The next step is to:

1. Upload the golden dataset as a LangSmith dataset.
2. Run `client.run_on_dataset(dataset_name, llm_or_chain, ...)` after
   each prompt change.
3. Use LangSmith's built-in evaluators (`criteria`, `qa`, `embedding_distance`)
   or a custom evaluator that checks `collected_data` against the expected dict.

This gives a persistent experiment history: every `prompt_config.yaml`
change produces a versioned eval run that can be compared to the baseline.

**Trigger:** Run the eval suite:
- Manually before any `prompt_config.yaml` change is merged.
- Automatically on a nightly schedule (not on every CI push — the cost
  of real LLM calls makes per-commit eval impractical).

---

## 2. Duplicate Detection Quality Evaluation

### What the current tests cover

The integration tests in `tests/integration/test_duplicate_detection.py`
use synthetic unit vectors (`EMBED_A`, `EMBED_B`, `EMBED_ORTHOGONAL`):

| Test | What it proves |
|---|---|
| Identical embedding → found | pgvector plumbing works |
| Orthogonal embedding → not found | Threshold logic works |
| Typo in request_type → found | pg_trgm plumbing works |
| Old row excluded by lookback | Date filter works |

**The gap:** These tests say nothing about accuracy on real-world data.
The `similarity_threshold: 0.85` in `app_config.yaml` was chosen without
empirical evidence. There is no measurement of precision or recall.

### Metrics to measure

**Precision** — of all requests flagged as duplicates, what fraction are
true duplicates?

```
Precision = True Positives / (True Positives + False Positives)
```

A low precision means users are interrupted with false warnings too often.

**Recall** — of all true duplicate pairs in the database, what fraction
does the pipeline detect?

```
Recall = True Positives / (True Positives + False Negatives)
```

A low recall means real duplicates are slipping through and being saved
as separate records.

**Per-stage contribution** — measuring precision/recall at the boundary
of each stage shows where the pipeline is doing useful work vs wasting
LLM calls.

| Stage boundary | Expected behaviour |
|---|---|
| After Stage 1 (pg_trgm) | High recall, low precision — wide net |
| After Stage 2 (pgvector) | Higher precision, recall maintained |
| After Stage 3 (LLM judge) | Highest precision, recall verified |

### Labelled test corpus

Create a set of (new_request, candidate_request, is_duplicate) triples
with human-assigned labels:

```json
[
  {
    "id": "dup-001",
    "new": {
      "request_type": "infrastructure-provisioning",
      "target_environment": "production",
      "business_justification": "Payment service Q4 scaling"
    },
    "candidate": {
      "request_type": "infrastructure-provisioning",
      "target_environment": "production",
      "business_justification": "Scaling infra for payment service peak load"
    },
    "is_duplicate": true,
    "notes": "Same intent, minor rephrasing"
  },
  {
    "id": "dup-002",
    "new": {
      "request_type": "infrastructure-provisioning",
      "target_environment": "production",
      "business_justification": "Payment service Q4 scaling"
    },
    "candidate": {
      "request_type": "infrastructure-provisioning",
      "target_environment": "development",
      "business_justification": "Payment service Q4 scaling"
    },
    "is_duplicate": false,
    "notes": "Different environment — not a duplicate"
  }
]
```

**Recommended minimum:** 100 labelled pairs (50 true duplicates,
50 non-duplicates).

### Threshold sensitivity sweep

Run precision/recall over a range of `similarity_threshold` values to
find the operating point that best fits the product's tolerance for
false positives vs false negatives:

```python
thresholds = [0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.95]
results = {}
for t in thresholds:
    tp = fp = fn = tn = 0
    for pair in labelled_corpus:
        predicted = run_pipeline(pair["new"], pair["candidate"], threshold=t)
        actual = pair["is_duplicate"]
        if predicted and actual:     tp += 1
        elif predicted and not actual: fp += 1
        elif not predicted and actual: fn += 1
        else:                        tn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    results[t] = {"precision": precision, "recall": recall}
```

This produces a precision-recall curve. The current threshold (0.85) can
be validated or adjusted based on real data rather than intuition.

### Stage 3 LLM judge isolation test

The judge (`_llm_judge_stage`) is currently only exercised in the E2E
mock path where it always returns `is_duplicate=True`. To evaluate it in
isolation:

```python
# tests/eval/test_duplicate_judge.py
pytestmark = pytest.mark.eval

@pytest.mark.parametrize("pair", LABELLED_PAIRS)
async def test_llm_judge_accuracy(pair, workflow):
    verdict = await workflow.llm_duplicate_judge.ainvoke([
        SystemMessage(content=DUPLICATE_JUDGE_PROMPT),
        AIMessage(content=(
            f"NEW REQUEST:\n{pair['new']}\n\n"
            f"EXISTING REQUEST:\n{pair['candidate']}"
        )),
    ])
    assert verdict.is_duplicate == pair["is_duplicate"], (
        f"Judge wrong on {pair['id']}: "
        f"got {verdict.is_duplicate}, "
        f"expected {pair['is_duplicate']}. "
        f"Reasoning: {verdict.reasoning}"
    )
```

### Embedding versioning concern

The `config_version` field is stored on every request row but is not
currently used during duplicate detection — Stage 2 compares embeddings
across all versions. If `prompt_config.yaml` changes significantly, old
embeddings (generated from v1.0 phrasing) may not be geometrically
comparable to new embeddings (generated from v2.0 phrasing).

**Planned mitigation:** Add a `config_version` filter to
`find_fuzzy_candidates` and `find_similar_requests` so Stage 1 and
Stage 2 only compare against records from the same schema version.
Stage 3 (LLM judge) is version-agnostic since it reads the field values,
not the embeddings.

---

## Summary

| Concern | Current gap | Planned fix | Effort |
|---|---|---|---|
| Task completion rate | Not measured | Golden dataset + pytest `--eval` | Medium |
| Extraction accuracy | Not measured | Field-level assertions in eval suite | Low |
| Guardrail robustness | Regex layer tested in unit suite; provider layer not measurable in isolation | Adversarial golden records covering both layers; note that novel injections not caught by regexes are handled by provider content policy (no additional app-level coverage needed) | Low |
| Duplicate precision/recall | Not measured | Labelled corpus + threshold sweep | Medium |
| Stage 3 judge accuracy | Mock only | Isolated judge eval against labelled pairs | Low |
| Embedding version drift | No mitigation | `config_version` filter in DB queries | Low |
