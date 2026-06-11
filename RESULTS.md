# TripMind — Evaluation Results

Three Llama 3.1 8B variants were trained using different supervision signals, then evaluated on 92 golden test cases from Indian domestic travel itineraries, plus 45 adversarial red-team prompts. An untuned `llama3.1:8b` baseline was added post-hoc to quantify training lift. The core question: **does richer teacher signal from agent reasoning traces produce a better travel optimizer than plain SFT on synthetic pairs?**

---

## The Four-Model Comparison

Each approach tested a different hypothesis:

- **llama3.1:8b (baseline)** — Untuned Llama 3.1 8B. Establishes the pre-training floor and confirms that fine-tuning was necessary for structured output compliance.
- **tripmind-ft** — Standard supervised fine-tuning on the 5,000 Phase 1 synthetic pairs (GPT-4o-mini teacher). Tests whether clean, validated <persona, optimized_itinerary> pairs are sufficient.
- **tripmind-distill** — Knowledge distillation from 500 Phase 2 DeepSeek agent reasoning traces. Tests whether exposing the model to multi-step tool-calling chains improves generalization.
- **tripmind-curriculum** — Two-stage curriculum: Phase 1 data first, then Phase 2 traces. Tests whether sequential training (domain knowledge before reasoning patterns) beats single-dataset approaches.

---

## Results (92 golden test cases × 4 models)

| Metric | Target | baseline | tripmind-ft | tripmind-distill | tripmind-curriculum |
|--------|--------|:--------:|:-----------:|:----------------:|:-------------------:|
| JSON valid | 85% | 0.0% ✗ | **100%** ✓ | 92.4% ✓ | 10.9% ✗ |
| Savings found | 70% | — ✗ | **100%** ✓ | 98.1% ✓ | — ✗ |
| Budget compliance | 80% | — ✗ | **98.7%** ✓ | — | — |
| Schema compliance | 80% | 0.0% ✗ | **83.7%** ✓ | 0.0% ✗ | 0.0% ✗ |
| Intent alignment | 55% | — | 32.2% ✗ | — | 41.8% ✗ |
| ROUGE-L vs teacher | 25% | 12.6% ✗ | **43.6%** ✓ | 8.9% ✗ | 12.7% ✗ |
| BERTScore F1 | 70% | 80.5% ✓ | **93.2%** ✓ | 73.8% ✓ | 73.4% ✓ |
| Reasoning coherence | 65% | — | **72.3%** ✓ | 67.4% ✓ | 47.0% ✗ |
| Grounding accuracy | 60% | — | 89.5% ✓ | 44.2% ✗ | **88.0%** ✓ |
| Red-team pass | 80% | — | 53.3% ✗ | 46.7% ✗ | **60.0%** ✗ |

### Head-to-head win rates (same 92 records, LLM judge)

- ft vs distill: **ft wins 78%** (72/92)
- ft vs curriculum: **ft wins 57%** (52/92)
- distill vs curriculum: distill wins 52% (48/92) — essentially tied

---

## Key Findings

### 0. Baseline confirms training was necessary

The untuned `llama3.1:8b` produced 0% valid JSON across all 92 test cases — it writes fluent natural language itineraries rather than structured output. Every structural metric (savings found, budget compliance, schema compliance) is unmeasurable as a result. This is the expected behavior of an instruction-tuned base model that has never seen the task's output schema.

Interestingly, the baseline scores 0.805 BERTScore — above the 0.70 target — despite producing no valid JSON. This reveals a **BERTScore blind spot**: semantic embedding similarity rewards natural language that mentions the right cities and concepts, even if the output is completely unstructured. ROUGE-L (baseline: 0.126 vs ft: 0.436) is the more honest differentiator here, because n-gram overlap correctly penalizes outputs that don't match the reference format. The baseline ROUGE-L failing while BERTScore passes is a diagnostic finding worth knowing before deploying embedding-based metrics for structured-output tasks.

### 1. Fine-tuning dominated on structural correctness

`tripmind-ft` achieved 100% JSON validity and 100% savings-finding rate — the two metrics most directly tied to the task contract. Training on clean, validated pairs taught the model exactly which output schema to produce. The distillation hypothesis didn't pan out: exposing the model to free-form reasoning chains during distillation may have introduced output-format noise that hurt structural compliance.

### 2. Curriculum training catastrophically broke JSON generation

The curriculum model's 10.9% JSON validity rate is the most striking result. Despite being initialized from Phase 1 SFT weights (which produced 100% valid JSON when trained independently), the Phase 2 fine-tuning stage appears to have overwritten structured-output behavior. The Phase 2 traces include long multi-agent reasoning chains — the model learned to generate those instead of compact JSON. This is a well-known failure mode in curriculum learning: the second stage can unlearn skills from the first.

### 3. Curriculum grounding is surprisingly competitive

Despite near-zero JSON validity, the curriculum model achieved 88.0% grounding accuracy (vs 89.5% for ft) — essentially tied. This suggests the Phase 2 reasoning traces *did* transfer real-world knowledge about Indian cities, transit options, and hotel pricing. The model knows the right answers; it just can't format them. A JSON-constrained decoding layer (e.g. Outlines or llama.cpp grammar sampling) would likely recover most of this value.

### 4. All models fail red-team robustness

None of the three models reached the 0.80 red-team pass target (ft: 0.53, distill: 0.47, curriculum: 0.60). The adversarial cases — budget bypass, constraint violations, prompt injection — exposed a consistent gap between supervised fine-tuning and adversarial robustness. This is expected: SFT optimizes for in-distribution outputs, not for rejecting out-of-distribution instructions. RLHF with adversarial preference data would be the right next step.

### 5. Intent alignment is low across all models

All three models scored well below the 0.55 intent alignment target. This metric measures whether the optimized itinerary matches the traveler's stated intents (Adventure, Nightlife, etc.) using sentence-transformers cosine similarity against the teacher output. The models consistently optimize for cost minimization while under-expressing the activity-level personalization. This points to a weakness in the Phase 1 training data, where cost savings were the primary reward signal.

---

## What I'd Do Differently

**Fix curriculum JSON:** Use grammar-constrained decoding (llama.cpp's `--grammar` flag or Outlines) during curriculum Phase 2, or add a JSON-only fine-tuning warmup after Phase 2 completes.

**Address red-team robustness:** Generate a 500-case adversarial dataset and apply Direct Preference Optimization (DPO) with the safe response as chosen and the unsafe response as rejected.

**Improve intent alignment:** Augment Phase 1 prompts to explicitly weight activity-level personalization alongside cost savings in the pivot analysis.

**Larger eval set:** 92 test cases is marginal. 500+ would give tighter confidence intervals on the ~57% ft vs curriculum head-to-head (currently not statistically significant at that margin).

---

## Training Efficiency

| Model | Final Train Loss | Steps | Hardware | Time |
|-------|:----------------:|:-----:|:--------:|:----:|
| tripmind-ft | **0.266** | 636 | Colab T4 | ~1.8h |
| tripmind-distill | 0.429 | 285 | Lightning.ai A100 | ~2.8h |
| tripmind-curriculum | 0.313 | 424+171 | Lightning.ai A100 | ~3.5h |

`tripmind-ft` had the lowest loss on the fewest steps — clean, consistent training pairs are highly sample-efficient. The distillation loss was highest (0.429), which correlates with the noisier output schema observed at eval time. The distillation hypothesis — that richer teacher signal would compensate for fewer training examples — did not hold at this scale (449 distillation pairs vs 4,749 fine-tuning pairs).
