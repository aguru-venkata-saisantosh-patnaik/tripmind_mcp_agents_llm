# TripMind — Engineering & Execution Notes

Real problems encountered and how they were solved, phase by phase.

---

## Phase 1 — Synthetic Data Generation

**Problem: GPT-4o-mini hallucinated prices outside budget tiers.**  
The teacher model occasionally produced itineraries where the per-person daily cost was outside the traveler's stated budget tier. A naive pipeline would silently include these, poisoning the training data.  
**Fix:** Built a 3-gate validator (`phase1_data_engine/validate.py`) that checks (1) valid JSON structure, (2) cost within ±20% of the budget tier bounds, and (3) savings ≥5% in the pivot. Rejection rate was ~12%. The pipeline retried failed records until 5,000 passed all three gates — ensuring the training data was internally consistent.

**Problem: Checkpoint safety for a long-running API job.**  
Generating 5,000 records at ~1.5 req/s takes hours. A crash mid-run would waste API spend.  
**Fix:** The generator writes each validated record immediately to disk in append mode and loads the existing output file on startup to skip already-generated IDs. Re-running the script after a crash picks up exactly where it left off.

---

## Phase 2 — Multi-Agent System

**Problem: Amadeus Flight API required a paid production key after the sandbox expired.**  
The original design used Amadeus for real flight pricing. Sandbox tokens have a 30-day TTL and free-tier quotas that expire without warning.  
**Fix:** Replaced Amadeus entirely with Overpass API (OpenStreetMap) for hotel and POI data, and with a haversine + mode-of-transport heuristic for transit cost estimation. No API key required. This turned a $0 sandbox into a genuinely free, rate-limit-free data source.

**Problem: DeepSeek's function-calling schema differs slightly from OpenAI's spec.**  
The MCP adapter was written for OpenAI tool-call format. DeepSeek returns `tool_calls` in the same structure but occasionally omits the `id` field or nests arguments differently.  
**Fix:** Added a normalisation layer in `mcp_adapter.py` that handles missing `id` fields and argument parsing inconsistencies before passing tool results back to the agent chain.

**Problem: MCP server port conflicts on the local machine.**  
Running 4 MCP servers + the agent pipeline simultaneously caused occasional `Address already in use` errors when a previous run hadn't fully exited.  
**Fix:** Added explicit port constants in `config.py` (`MCP_SERVERS` dict) and documented the `lsof -ti:800x | xargs kill` cleanup command in the Phase 2 README.

**Problem: Agent traces had variable quality — some agents looped or produced empty tool calls.**  
About 8% of traces had malformed tool call sequences where the agent repeated the same call in a loop or returned an empty JSON body.  
**Fix:** Added a quality filter in the trace collection pipeline: traces with >6 tool calls per agent, duplicate consecutive tool calls, or empty JSON in the final `optimizer` response were discarded. This dropped the raw 545 traces to 500 clean ones.

---

## Phase 3 — SLM Training

**Problem: Colab T4 OOM with seq_len=512 and batch_size=4.**  
The initial training config mirrored Phase 2's seq_len=16384 (designed for A100). The T4 has 15GB VRAM — it OOM'd immediately.  
**Fix:** Reduced seq_len to 512 (sufficient for Phase 1 pairs, which are compact JSON), used fp16 (not bf16), and gradient accumulation of 4 to maintain effective batch size. Final config fit in ~11GB.

**Problem: Curriculum model Phase 2 training catastrophically overwrote Phase 1 JSON behaviour.**  
After Stage 1 (Phase 1 SFT), the model produced valid JSON. After Stage 2 (Phase 2 traces), it produced multi-paragraph reasoning chains and 10.9% valid JSON. This was a curriculum learning failure — the second stage's signal distribution was too different from Stage 1.  
**Fix:** This was documented as a finding rather than fixed — fixing it would have required a JSON-constrained grammar layer during Stage 2 decoding or a brief JSON-only warmup after Stage 2, which was beyond the project's compute budget. The curriculum model is now recommended for use with grammar-constrained decoding only.

**Problem: GGUF conversion required llama.cpp built from source.**  
HuggingFace's standard export pipeline doesn't produce GGUF directly; `convert_hf_to_gguf.py` from llama.cpp is required.  
**Fix:** Built llama.cpp from source in the training notebooks and used the `Q4_K_M` quantisation level, which balances quality and size (4.6 GB per model). Unsloth's `model.save_pretrained_gguf()` shortcut worked correctly on the A100 notebooks and is the recommended path.

---

## Phase 4 — Evaluation

**Problem: Gemini 2.0 Flash rate-limited the judge calls mid-eval.**  
The eval pipeline calls Gemini once per record per model for reasoning coherence and grounding accuracy. At 92 × 3 = 276 judge calls in rapid succession, Gemini's free tier returned 429s after ~80 calls.  
**Fix:** Added a 2-second sleep between judge calls and a retry-with-backoff decorator in `phase4_evals/utils.py`. Total eval wall time increased from ~25 minutes to ~55 minutes but succeeded without errors.

**Problem: BERTScore gave the untuned baseline a misleadingly high score.**  
The baseline model (llama3.1:8b, 0% JSON validity) scored 0.805 BERTScore — above the 0.70 target — because natural language about Indian cities is semantically close to the reference JSON. This would have made the baseline look competitive if BERTScore were the only metric.  
**Fix:** Not a code fix — a methodology finding. ROUGE-L (0.126 for baseline vs 0.436 for ft) correctly penalises format mismatch and was added as the primary structural-proximity metric. The BERTScore blind spot is documented in RESULTS.md.

---

## Phase 5 — FastAPI Server

**Problem: starlette version conflict between the MCP library and FastAPI.**  
The project's base conda environment installs `starlette==1.2.1` as a transitive dependency of the `mcp` library. FastAPI 0.116+ requires starlette ≥0.27. The two are mutually incompatible in the same environment.  
**Fix:** Consolidated all dependencies into a single root `requirements.txt`. Pip resolves the starlette version automatically when installing into a clean environment (e.g. a fresh venv or Colab). The conflict only manifests when installing into an environment that already has the `mcp` library pre-installed at an incompatible starlette version — using `pip install -r requirements.txt` in a clean environment avoids the issue entirely.

---

## Cross-Cutting

**Problem: City coordinates and haversine distance duplicated across two MCP servers.**  
`routing_server.py` and `hotels_server.py` each had an identical 20-city coordinate dictionary and an identical `_haversine_km()` function defined locally.  
**Fix:** Centralised city coordinates into `config.CITY_COORDS` and the haversine function into `utils/geo.py`. Both servers now import from the shared locations.

**Problem: Data files too large for GitHub (88MB training JSONL, 37MB traces, 14MB synthetic).**  
A clean GitHub push requires keeping repo size manageable; large generated files add no value since they're reproducible.  
**Fix:** `.gitignore` excludes `data/synthetic/`, `data/traces/`, `data/training/`, `data/seeds/`, and all model GGUFs. Only `data/evals/` is committed — the 92-case golden set, eval results, and charts are small (< 3MB total) and are the primary portfolio evidence.
