# MedSoft Assesment

---

## Before you start

You need two API keys:

- **SerpAPI** — used to run web and shopping searches. Get one at [serpapi.com](https://serpapi.com)
- **Gemini** — used to extract product info from pages. Get one at [aistudio.google.com](https://aistudio.google.com)

Create a `.env` file in the project root and add both keys:

```
serp_api_key=your_serpapi_key_here
genai_api_key=your_gemini_key_here
```

---

## Running locally

Make sure you have Python 3.11+ installed.

**1. Create a virtual environment and install dependencies:**

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

**2. Start the server:**

```bash
python main.py
```

The app will be running at `http://localhost:8000`.

Open that in your browser and you'll see a simple search form where you can test it directly.

---

## Running with Docker

**1. Build the image:**

```bash
docker build -t product-search .
```

**2. Run the container (passing in your `.env` file):**

```bash
docker run --env-file .env -p 8000:8000 product-search
```

Same as before — open `http://localhost:8000` in your browser.

---

## API usage

Send a POST request to `/search`:

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "dell laptop", "region": "UAE"}'
```

**Supported regions:**

Saudi, Saudi Arabia, Egypt, USA, United States, UAE, United Arab Emirates, Qatar, Kuwait, Bahrain, Oman, Jordan, United Kingdom, UK, Canada, Germany, France, India

**Response format:**

```json
{
  "results": [
    {
      "product_name": "Dell Inspiron 15",
      "price": 2999.0,
      "currency": "AED",
      "availability": "in_stock",
      "source": "noon.com",
      "source_url": "https://www.noon.com/...",
      "relevance_score": 0.75
    }
  ]
}
```

---

## Logs

The app logs to both the console and a file called `app.log` in the project root. If something isn't working, that's the first place to check.

Thanks, Yassin.
