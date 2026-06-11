# TripMind

Autonomous multi-agent AI travel optimizer that finds **Price-Pivot Points** — transit, accommodation, and activity substitutions that save ≥5% without degrading trip quality.

Built for Indian domestic travel. Trains three SLMs (fine-tuned, distilled, curriculum-trained) and compares them — testing whether richer teacher signal from agent reasoning traces improves generalization over plain SFT on synthetic data.

---

## Phase Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Synthetic data engine (5,000 pairs, gpt-4o-mini) | ✅ Complete |
| 2 | Multi-agent orchestration + MCP servers (500 traces, DeepSeek) | ✅ Complete |
| 3 | Three SLMs trained — Colab T4 (ft) + Lightning.ai A100 (distill, curriculum) | ✅ Complete |
| 4 | Evaluation suite + red teaming (92 test cases × 3 trained + 1 baseline + 45 red team) | ✅ Complete |
| 5 | FastAPI inference server — REST endpoints for all 4 models | 🔲 In Progress |

---

## Tech Stack

| Component | Technology | Cost |
|-----------|-----------|------|
| Phase 1 teacher | OpenAI gpt-4o-mini | Paid (~$1.50 / 5k records) |
| Phase 2 agent LLM | DeepSeek (`deepseek-chat`) | Free (5M tokens) |
| Phase 4 eval judge | Gemini 2.0 Flash (AI Studio) | Free (1M tokens/day) |
| SLM base model | Llama 3.1 8B (Unsloth + QLoRA r=8) | Free |
| SLM training (ft) | Google Colab T4, fp16, seq_len=512 | Free |
| SLM training (distill, curriculum) | Lightning.ai A100, bf16, seq_len=16384 | Free (3h credit) |
| SLM inference | Ollama (local, GGUF Q4_K_M 4.6GB each) | Free |
| Routing | OpenRouteService + Nominatim | Free |
| Hotels / Flights | Overpass API (OSM) + haversine | Free |
| POIs / Restaurants | Overpass API (OSM) | Free |
| Web search | duckduckgo-search library | Free |
| Intent alignment | sentence-transformers (local) | Free |
| Inference API | FastAPI + Uvicorn | Free |

---

## Project Structure

```
travel_project/
├── config.py                        # all shared constants — never hardcode anything
├── requirements.txt
├── utils/
│   ├── logger.py                    # structured JSON logger (all phases)
│   └── cache.py                     # disk-based API response cache
├── phase1_data_engine/              # ✅ COMPLETE
│   ├── generate.py                  # async gpt-4o-mini pipeline, checkpoint-safe
│   ├── validate.py                  # 3-gate validator
│   └── schemas.py
├── phase2_agents/                   # ✅ COMPLETE
│   ├── mcp_servers/                 # 4 MCP servers (SSE transport)
│   │   ├── routing_server.py        # port 8001 — OpenRouteService + Nominatim
│   │   ├── hotels_server.py         # port 8002 — Overpass hotels + haversine flights
│   │   ├── overpass_server.py       # port 8003 — OSM POIs + restaurants
│   │   └── search_server.py         # port 8004 — DuckDuckGo
│   ├── agents/
│   │   ├── analyst.py               # transit + hotel cost analysis
│   │   ├── concierge.py             # POI + dining substitutions
│   │   └── optimizer.py             # final itinerary synthesis
│   ├── mcp_adapter.py               # MCP → OpenAI-compatible tool bridge
│   ├── supervisor.py                # orchestrates the 3-agent chain
│   ├── llm_utils.py                 # async LLM wrapper (DeepSeek-compatible)
│   ├── llm_client.py                # provider switcher (deepseek / groq)
│   └── run.py                       # CLI entrypoint
├── phase3_training/                 # ✅ COMPLETE
│   ├── prepare_ft.py                # Phase 1 → Alpaca (SFT)
│   ├── prepare_distill.py           # Phase 2 → Alpaca (distillation)
│   ├── prepare_curriculum.py        # stage1 + stage2 files
│   ├── verify_datasets.py           # pre-upload sanity check
│   └── notebooks/
│       ├── 01_train_ft.ipynb        # Colab: tripmind-ft (3 epochs)
│       ├── 02_train_distill.ipynb   # Colab: tripmind-distill (5 epochs)
│       ├── 03_train_curriculum.ipynb# Colab: tripmind-curriculum (2-stage)
│       └── modelfiles/              # Ollama Modelfile for each model
├── phase4_evals/                    # ✅ COMPLETE — eval suite + red team
├── phase5_serving/                  # 🔲 IN PROGRESS — FastAPI inference API
├── data/
│   ├── synthetic/                   # Phase 1 — 5,000 validated pairs
│   ├── traces/                      # Phase 2 — 500 quality agent traces (1 file)
│   ├── training/                    # Phase 3 — 6 Alpaca JSONL files
│   ├── evals/                       # Phase 4 — eval + red team results
│   └── seeds/                       # 50k seed personas
├── models/
│   ├── finetune/                    # tripmind-ft.gguf (4.6GB, Q4_K_M)
│   ├── distill/                     # tripmind-distill.gguf (4.6GB, Q4_K_M)
│   └── curriculum/                  # tripmind-curriculum.gguf (4.6GB, Q4_K_M)
└── logs/phase1/ … phase4/
```

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Fill in: DEEPSEEK_API_KEY, GOOGLE_API_KEY, ORS_API_KEY
```

---

## Phase 2 — Running Agents

```bash
# Start all 4 MCP servers (4 separate terminals)
python phase2_agents/mcp_servers/routing_server.py   # port 8001
python phase2_agents/mcp_servers/hotels_server.py    # port 8002
python phase2_agents/mcp_servers/overpass_server.py  # port 8003
python phase2_agents/mcp_servers/search_server.py    # port 8004

# Run agent pipeline
python phase2_agents/run.py --limit 25 --concurrency 3 --verbose
```

---

## Phase 3 — Register Models with Ollama

GGUFs are already in `models/`. Run from the **project root**:

```bash
ollama create tripmind-ft         -f phase3_training/notebooks/modelfiles/Modelfile.ft
ollama create tripmind-distill    -f phase3_training/notebooks/modelfiles/Modelfile.distill
ollama create tripmind-curriculum -f phase3_training/notebooks/modelfiles/Modelfile.curriculum

ollama list
ollama run tripmind-ft "Persona: Solo, Delhi to Goa, Budget+, 5 days. Optimize."
```

HuggingFace backups (LoRA adapters + GGUF):
- `agurusantosh/tripmind-ft-lora` / `agurusantosh/tripmind-ft-gguf`
- `agurusantosh/tripmind-distill-lora` / `agurusantosh/tripmind-distill-gguf`
- `agurusantosh/tripmind-curriculum-lora` / `agurusantosh/tripmind-curriculum-gguf`

---

## Phase 4 — Eval Results

92 test cases × 4 models (3 trained + untuned baseline) + 45 adversarial red-team prompts. Full narrative: [`RESULTS.md`](RESULTS.md).

| Metric | Target | baseline | tripmind-ft | tripmind-distill | tripmind-curriculum |
|--------|:------:|:--------:|:-----------:|:----------------:|:-------------------:|
| JSON valid | 85% | 0.0% ✗ | **100%** ✓ | 92.4% ✓ | 10.9% ✗ |
| Savings found | 70% | — ✗ | **100%** ✓ | 98.1% ✓ | — |
| Budget compliance | 80% | — ✗ | **98.7%** ✓ | — | — |
| Schema compliance | 80% | 0.0% ✗ | **83.7%** ✓ | 0.0% ✗ | 0.0% ✗ |
| ROUGE-L | 25% | 12.6% ✗ | **43.6%** ✓ | 8.9% ✗ | 12.7% ✗ |
| BERTScore F1 | 70% | 80.5%\* ✓ | **93.2%** ✓ | 73.8% ✓ | 73.4% ✓ |
| Grounding accuracy | 60% | — | 89.5% ✓ | 44.2% ✗ | **88.0%** ✓ |
| Red-team pass | 80% | — | 53.3% ✗ | 46.7% ✗ | **60.0%** ✗ |

\* Baseline BERTScore passes the target despite 0% JSON validity — see RESULTS.md for why BERTScore is a weak signal for structured-output tasks.

**Head-to-head:** ft beats distill 78%, ft beats curriculum 57%.

![Radar chart](data/evals/charts/radar_all_metrics.png)

![Structural vs semantic](data/evals/charts/structural_vs_semantic.png)

![Head-to-head win rates](data/evals/charts/head_to_head.png)

![Red-team pass rates](data/evals/charts/red_team_pass.png)

---

## Phase 5 — Inference API

```bash
# Install (use a fresh venv — base env has conflicting MCP/starlette versions)
python -m venv .venv-serving && source .venv-serving/bin/activate
pip install -r phase5_serving/requirements.txt

# Start
uvicorn phase5_serving.api.main:app --reload --port 8000

# Docs
open http://localhost:8000/docs
```

```bash
# Example: optimize a persona
curl -X POST http://localhost:8000/optimize \
  -H "Content-Type: application/json" \
  -d '{"model": "tripmind-ft", "persona": {"starting_city": "Mumbai",
       "destination_city": "Delhi", "type": "Solo",
       "size": {"adults": 1, "children": 0}, "intents": ["Adventure"],
       "budget": "Shoestring", "duration_days": 5, "duration_nights": 4}}'
```

---

## Data Flow

```
data/seeds/persona_seeds_50k.jsonl
        │ Phase 1 (gpt-4o-mini)
        ▼
data/synthetic/v2_20260608_085742.jsonl     (5,000 validated pairs)
        │ Phase 2 (DeepSeek + MCP tools)
        ▼
data/traces/agent_traces_all.jsonl          (500 quality reasoning traces)
        │ Phase 3 (Colab T4 + Unsloth QLoRA)
        ▼
models/finetune/    → tripmind-ft           (SFT on Phase 1)
models/distill/     → tripmind-distill      (distilled from Phase 2)
models/curriculum/  → tripmind-curriculum   (Phase 1 → Phase 2, sequential)
        │ Phase 4 (Groq judge + sentence-transformers)
        ▼
data/evals/eval_results_*.jsonl             (92×3 evals + 45 red team)
        │ Phase 5 (FastAPI)
        ▼
phase5_serving/api: REST inference endpoints (GET /health, GET /models, POST /optimize)
```

---

## Key Design Decisions

**Why three training approaches?** Testing a research question: does distilling DeepSeek's agent reasoning chains produce a better travel optimizer than plain SFT on synthetic pairs? Curriculum training tests whether sequential training (domain first, reasoning second) beats both single-dataset approaches. The three-way comparison is the core portfolio contribution.

**Why DeepSeek for Phase 2?** OpenAI-compatible API, strong function-calling, generous free tier (5M tokens). Its 500 reasoning traces are the distillation training signal for Phase 3.

**Why Groq for Phase 4 judging?** Free tier, fast inference via Llama 3.1 8B. Intent alignment uses sentence-transformers locally (no API calls needed).

**Why MCP servers?** Standard protocol using the official `mcp` Python library (SSE transport). Same servers plug directly into Claude Desktop/Code without modification.

**Why cache API responses?** ~20 unique city pairs collapses 500 agent runs to ~40 real routing API calls — well within OpenRouteService's 2,000/day free limit.
