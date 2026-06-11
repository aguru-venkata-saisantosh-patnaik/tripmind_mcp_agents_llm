# TripMind Inference API

FastAPI server exposing the four TripMind models via REST.

## Setup

```bash
# From the project root — all dependencies are in the central requirements.txt
pip install -r requirements.txt
```

## Start the server

```bash
uvicorn phase5_serving.api.main:app --reload --port 8000
```

Interactive docs: **http://localhost:8000/docs**

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Ollama ping + model availability |
| `GET` | `/models` | List all 4 registered TripMind models |
| `POST` | `/optimize` | Run inference on a persona dict |
| `GET` | `/results/summary` | Latest eval summary JSON (instant) |
| `GET` | `/results/compare` | Head-to-head win rates |

## Example

```bash
curl -X POST http://localhost:8000/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tripmind-ft",
    "persona": {
      "starting_city": "Mumbai",
      "destination_city": "Delhi",
      "type": "Solo",
      "size": {"adults": 1, "children": 0},
      "intents": ["Adventure"],
      "budget": "Shoestring",
      "duration_days": 5,
      "duration_nights": 4
    }
  }'
```

## Environment variable

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_TIMEOUT` | `600` | Seconds before Ollama call times out (CPU inference is slow) |
