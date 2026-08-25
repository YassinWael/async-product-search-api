# Async Product Search API

[![tests](https://github.com/YassinWael/async-product-search-api/actions/workflows/tests.yml/badge.svg)](https://github.com/YassinWael/async-product-search-api/actions/workflows/tests.yml)

A FastAPI service that searches for products across regional shopping results
and public product pages, maps inconsistent source data into one Pydantic
schema, removes duplicates, and ranks the final records.

Direct product pages can be parsed deterministically, while irregular and
listing pages can use optional Gemini extraction. Model output is
schema-validated before it enters the result pipeline.

## What it demonstrates

- Async HTTP collection with timeouts, bounded retry/backoff, and a bounded TTL cache
- Regional Google Shopping and organic-search fallback paths
- HTML title, price, regional currency, availability, and product-link extraction
- Optional structured Gemini extraction through a Pydantic response schema
- Concurrent source processing with normalized and deduplicated results
- FastAPI request/response validation and a small browser interface
- Network-free tests, Docker packaging, and GitHub Actions verification

## Flow

```text
POST /search
    -> validate query and region
    -> Google Shopping where supported
       -> fetch per-product store offers
    -> otherwise fall back to organic results
       -> parse direct product HTML or extract irregular pages with Gemini
    -> resolve listing pages to concrete product URLs
    -> normalize, deduplicate, rank, and return JSON
```

## Setup

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
copy .env.example .env
```

Add a SerpAPI key to `.env`. A Gemini key is optional:

```dotenv
SERP_API_KEY=replace_me
GEMINI_API_KEY=replace_me
```

Start the API:

```bash
uvicorn main:app --reload
```

Then open `http://localhost:8000`, or call the endpoint directly:

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"dell laptop","region":"Saudi Arabia"}'
```

Example response:

```json
{
  "results": [
    {
      "product_name": "Dell Inspiron 15",
      "price": 2999.0,
      "currency": "SAR",
      "availability": "in_stock",
      "source": "Example Store",
      "source_url": "https://example.com/product/dell-inspiron-15",
      "relevance_score": 0.5
    }
  ]
}
```

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests use fake HTTP responses and local HTML strings. They never require API
keys or make external requests.

The included browser page is a local demo. Add authentication and a distributed
rate limiter before exposing the API directly to the internet.

## Docker

```bash
docker build -t async-product-search .
docker run --env-file .env -p 8000:8000 async-product-search
```

Licensed under the [MIT License](LICENSE).
