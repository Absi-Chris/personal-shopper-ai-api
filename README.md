# Personal Shopper AI Assistant API

FastAPI + LangChain agent + OpenAI tool calling + SQLite catalog/cart + Chroma RAG.

## 1. Setup

Python 3.11+ recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # Windows: copy .env.example .env
```

Add your OpenAI API key to `.env`.

## 2. Initialize and run

```bash
python -m scripts.seed
uvicorn app.main:app --reload
```

Open http://localhost:8000/docs

## 3. Example requests

Create a product:
```bash
curl -X POST http://localhost:8000/products \
  -H "Content-Type: application/json" \
  -d '{"sku":"CAM-001","name":"Mirrorless Camera","brand":"Canon","category":"Cameras","price":35000,"stock":4,"rating":4.8,"description":"Compact mirrorless camera"}'
```

Chat with the shopper:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-User-Id: demo-user" \
  -d '{"message":"Find headphones under 3000 pesos and compare the best options"}'
```

Add to cart through API:
```bash
curl -X POST http://localhost:8000/cart/items \
  -H "Content-Type: application/json" \
  -H "X-User-Id: demo-user" \
  -d '{"product_id":1,"quantity":2}'
```

View cart:
```bash
curl http://localhost:8000/cart -H "X-User-Id: demo-user"
```

Upload a policy, manual or product document:
```bash
curl -X POST http://localhost:8000/documents \
  -F "file=@warranty-policy.pdf"
```

Original uploads are saved in `./documents` by default, while extracted text is indexed in Chroma under `./chroma_data`. Set `DOCUMENTS_DIR` to change the upload folder. Docker Compose persists uploads in the workspace's `documents/` directory.

Then ask:
```json
{"message":"What is the warranty coverage and how do I claim it?"}
```

## 4. API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/products` | Create catalog product |
| GET | `/products` | Filter/search catalog |
| POST | `/cart/items` | Add cart item |
| GET | `/cart` | View cart and total |
| DELETE | `/cart/items/{product_id}` | Remove cart item |
| POST | `/chat` | AI shopping assistant |
| GET | `/documents` | List saved RAG documents |
| POST | `/documents` | Upload and index TXT, MD, PDF |
| DELETE | `/documents/{stored_as}` | Delete saved document and indexed chunks |

## 5. Agent tools

The LangChain agent has six tools:
- `search_products`: SQLite keyword, category and price filtering.
- `compare_products`: fetches selected product records for comparison.
- `add_to_cart`: checks stock and updates the active user's cart.
- `remove_from_cart`: removes a product from the active cart.
- `view_cart`: returns cart lines and totals.
- `ask_product_documents`: semantic retrieval from uploaded documents.

The agent uses LangChain's `create_agent` tool loop. Tool schemas are exposed to the OpenAI chat model, which selects tools based on the user's request.

## 6. Evaluation and guardrails

The live agent uses Guardrails AI input validators (`DetectJailbreak`, `RestrictToTopic`) before any agent or cart-tool execution, and output validators (`ProfanityFree`, `GuardrailsPII`) before returning a response. `CompetitorCheck` is enabled when you configure `GUARDRAIL_COMPETITORS` as a comma-separated list; it is configurable because this catalog intentionally compares multiple brands. The app also normalizes the rupee symbol to `₱` and refuses to confirm a cart mutation unless a tool result confirms success.

Guardrails validators download local model assets. Install the project requirements, then run each validator's setup step once:
```bash
.venv/bin/python -m guardrails_ai.detect_jailbreak.post_install
.venv/bin/python -m guardrails_ai.competitor_check.post_install
.venv/bin/python -m guardrails_ai.profanity_free.post_install
.venv/bin/python -m guardrails_ai.guardrails_pii.post_install
```
`RestrictToTopic` downloads its zero-shot classifier the first time it runs. The output PII validator is configured to reject responses containing detected names, email addresses, phone numbers, or locations. Set `GUARDRAIL_COMPETITORS` to a comma-separated list to enable competitor filtering; leave it empty for the multi-brand sample catalog. Validation errors fail closed with a safe response. These validators use local ML models and may add startup latency to the first chat request; ensure the model downloads complete before serving traffic.

LangChain model responses are cached in SQLite at `./data/llm_cache.db` by default, so identical prompts can reuse earlier completions across restarts. Configure `LLM_CACHE_PATH` to change the location or set `LLM_CACHE_ENABLED=false` to disable it. The cache stores prompts and model responses in plaintext; protect the file accordingly. RAGAS metric-judge calls use its own OpenAI client and are not covered by this LangChain cache.

To run the offline guardrail tests:
```bash
python -m unittest discover -s tests
```

RAGAS scoring evaluates faithfulness, answer relevancy, context precision and context recall against a JSONL golden set. Upload the policy/manual documents first, then create `evals/rag_cases.jsonl` with one JSON object per line:
```json
{"question":"What is the warranty period?","reference_answer":"The warranty lasts 12 months."}
```

Run the evaluation with your configured OpenAI API key:
```bash
python3 -m venv .eval-venv
.eval-venv/bin/pip install -r requirements-eval.txt
.eval-venv/bin/python -m scripts.evaluate_rag --dataset evals/rag_cases.jsonl --output evals/rag_results.json
```

Results are saved to `evals/rag_results.json` by default. Override the destination with `--output path/to/results.json`. RAGAS uses model-based scoring and can incur API costs. Its scores should be treated as regression signals, not as a substitute for reviewing representative conversations.

## 7. Important implementation notes

- Product prices and stock are read from SQLite, not generated by the model.
- Cart ownership is scoped by `X-User-Id`; this is a demo identity header, **not authentication**. Replace it with a verified JWT/OAuth subject before production.
- `/chat` persists the last 12 messages per `session_id` and `user_id` in SQLite, providing basic multi-turn conversation context. For production, add session expiry, ownership checks, retention controls and/or a LangGraph checkpointer.
- SQLite is suitable for a local prototype or low-concurrency workload. Use PostgreSQL for a busy multi-user deployment.
- Chroma persists document vectors separately from SQLite. Keep `chroma_data` on persistent storage.
- PDF ingestion extracts text only; scanned PDFs need OCR.
- Add admin authorization to product/document write endpoints, upload malware scanning, rate limits, request size limits, and audit logs before production.
- `FAISS` can be substituted for Chroma if an in-process vector index is preferred. Chroma is used here for convenient persistent local storage.
- Validate the installed LangChain package versions together in a lockfile before deployment.
