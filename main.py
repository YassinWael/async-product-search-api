import asyncio
import logging
import os
import re
import sys
import time
from contextlib import asynccontextmanager
from urllib.parse import unquote, urljoin, urlparse

import aiohttp
import uvicorn
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from google import genai
from pydantic import BaseModel

from models import ProductResult, SearchRequest, SearchResponse

load_dotenv()

SERP_API_KEY = os.getenv("serp_api_key")
GENAI_API_KEY = os.getenv("genai_api_key")
SERPAPI_URL = "https://serpapi.com/search.json"
GEMINI_MODEL = "gemini-2.5-flash-lite"
CACHE_TTL_SECONDS = 300

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log"),
    ],
)
logger = logging.getLogger(__name__)

REGION_CONFIG = {
    "Saudi": {"location": "Riyadh, Saudi Arabia", "hl": "en", "gl": "sa"},
    "Saudi Arabia": {"location": "Riyadh, Saudi Arabia", "hl": "en", "gl": "sa"},
    "Egypt": {"location": "Cairo, Egypt", "hl": "en", "gl": "eg"},
    "USA": {"location": "United States", "hl": "en", "gl": "us"},
    "United States": {"location": "United States", "hl": "en", "gl": "us"},
    "UAE": {"location": "Abu Dhabi, United Arab Emirates", "hl": "en", "gl": "ae"},
    "United Arab Emirates": {"location": "Abu Dhabi, United Arab Emirates", "hl": "en", "gl": "ae"},
    "Qatar": {"location": "Doha, Qatar", "hl": "en", "gl": "qa"},
    "Kuwait": {"location": "Kuwait", "hl": "en", "gl": "kw"},
    "Bahrain": {"location": "Manama, Bahrain", "hl": "en", "gl": "bh"},
    "Oman": {"location": "Muscat, Oman", "hl": "en", "gl": "om"},
    "Jordan": {"location": "Amman, Jordan", "hl": "en", "gl": "jo"},
    "United Kingdom": {"location": "London, United Kingdom", "hl": "en", "gl": "gb"},
    "UK": {"location": "London, United Kingdom", "hl": "en", "gl": "gb"},
    "Canada": {"location": "Ottawa, Canada", "hl": "en", "gl": "ca"},
    "Germany": {"location": "Berlin, Germany", "hl": "de", "gl": "de"},
    "France": {"location": "Paris, France", "hl": "fr", "gl": "fr"},
    "India": {"location": "New Delhi, India", "hl": "en", "gl": "in"},
}

SUPPORTED_SHOPPING_GLS = {"ae", "ca", "de", "fr", "gb", "in", "sa", "us"}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_session
    http_session = aiohttp.ClientSession(headers={"User-Agent": "Mozilla/5.0"})
    yield
    await http_session.close()


app = FastAPI(title="Product Search API", lifespan=lifespan)
gemini_client = genai.Client(api_key=GENAI_API_KEY) if GENAI_API_KEY else None
http_session: aiohttp.ClientSession | None = None
http_cache = {}


class GeminiProductExtraction(BaseModel):
    product_name: str | None = None
    price: float | None = None
    currency: str | None = None
    availability: str | None = None


class GeminiPageExtraction(BaseModel):
    products: list[GeminiProductExtraction]


def make_cache_key(prefix, url, params=None):
    """
    Creates a cache key given a prefix, a URL, and optional parameters.

    :param prefix: The prefix to use for the cache key.
    :param url: The URL to use for the cache key.
    :param params: Optional parameters to use for the cache key.
    :return: A tuple representing the cache key.
    """
    normalized_params = tuple(sorted((params or {}).items()))
    return (prefix, url, normalized_params)


def get_cached_value(key):
    cached = http_cache.get(key)
    if not cached:
        return None

    expires_at, value = cached
    if expires_at < time.time():
        http_cache.pop(key, None)
        return None
    return value


def set_cached_value(key, value, ttl=CACHE_TTL_SECONDS):
    http_cache[key] = (time.time() + ttl, value)


async def fetch_text(url, *, params=None, timeout=20, headers=None):
    cache_key = make_cache_key("text", url, params)
    cached = get_cached_value(cache_key)
    if cached is not None:
        logger.info("cache hit text %s", url)
        return cached

    logger.info("fetch text %s", url)
    client_timeout = aiohttp.ClientTimeout(total=timeout)
    request_headers = headers or {"User-Agent": "Mozilla/5.0"}
    async with http_session.get(url, params=params, headers=request_headers, timeout=client_timeout) as response:
        response.raise_for_status()
        text = await response.text()
    set_cached_value(cache_key, text)
    return text


async def fetch_json(url, *, params=None, timeout=30, headers=None):
    cache_key = make_cache_key("json", url, params)
    cached = get_cached_value(cache_key)
    if cached is not None:
        logger.info("cache hit json %s", url)
        return cached

    logger.info("fetch json %s", url)
    client_timeout = aiohttp.ClientTimeout(total=timeout)
    request_headers = headers or {"User-Agent": "Mozilla/5.0"}
    async with http_session.get(url, params=params, headers=request_headers, timeout=client_timeout) as response:
        response.raise_for_status()
        data = await response.json(content_type=None)
    set_cached_value(cache_key, data)
    return data


def get_currency(price_text):
    if not price_text:
        return None
    if "SAR" in price_text:
        return "SAR"
    if "USD" in price_text or "$" in price_text:
        return "USD"
    if "AED" in price_text:
        return "AED"
    if "EGP" in price_text:
        return "EGP"
    return None


def get_availability(details):
    if not details:
        return None

    text = " ".join(details).lower()
    if "out of stock" in text or "not available" in text:
        return "out_of_stock"
    if "in stock" in text:
        return "in_stock"
    if "pre-order" in text or "preorder" in text:
        return "preorder"
    return None


def get_score(query, title):
    query_words = set(query.lower().split())
    title_words = set(title.lower().split())
    if not query_words:
        return 0.0
    return round(len(query_words & title_words) / len(query_words), 2)


def extract_price_and_currency(text):
    if not text:
        return None, None

    patterns = [
        (r"(SAR|AED|USD|EGP)\s*([0-9][0-9,\.]*)", False),
        (r"([0-9][0-9,\.]*)\s*(SAR|AED|USD|EGP)", True),
        (r"(ج\.?م|E£|LE)\s*([0-9][0-9,\.]*)", False),
        (r"([0-9][0-9,\.]*)\s*(ج\.?م|E£|LE)", True),
    ]

    for pattern, reversed_match in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue

        if reversed_match:
            amount_text, currency_text = match.groups()
        else:
            currency_text, amount_text = match.groups()

        amount_text = amount_text.replace(",", "")
        try:
            amount = float(amount_text)
        except ValueError:
            continue

        currency = get_currency(currency_text.upper())
        if currency_text in {"ج.م", "E£", "LE"}:
            currency = "EGP"
        return amount, currency

    return None, None


def is_obvious_bad_page(link, title):
    text = f"{link} {title}".lower()
    hints = ["/blog", "/blogs/", "/news/", "official site", "dubizzle", "olx", "/lp"]
    return any(hint in text for hint in hints)


def is_listing_like_page(link, title):
    text = f"{link} {title}".lower()
    hints = [
        "/category",
        "/categories",
        "/shop/",
        "/laptops",
        "/collections/",
        "/c/",
        "best prices",
        "shop ",
        "online",
    ]
    product_hints = ["/dp/", "/product/", "/products/", "/p/"]
    if any(hint in text for hint in product_hints):
        return False
    return any(hint in text for hint in hints)


async def scrape_page(link, fallback_title):
    try:
        html = await fetch_text(link, timeout=20)
    except aiohttp.ClientError:
        logger.warning("scrape failed %s", link)
        return None

    title_match = re.search(r'<meta[^>]+property="og:title"[^>]+content="([^"]+)"', html, re.IGNORECASE)
    if title_match:
        title = title_match.group(1)
    else:
        title_match = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        title = title_match.group(1).strip() if title_match else fallback_title

    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    price, currency = extract_price_and_currency(text)
    availability = get_availability([text])

    if price is None:
        return None

    return {
        "title": title,
        "price": price,
        "currency": currency,
        "availability": availability,
    }


async def extract_with_gemini(link, fallback_title, max_products):
    if not gemini_client:
        return []

    try:
        html = await fetch_text(link, timeout=20)
    except aiohttp.ClientError:
        logger.warning("gemini fetch failed %s", link)
        return []

    html = re.sub(r"<script.*?</script>", " ", html, flags=re.IGNORECASE | re.DOTALL)
    html = re.sub(r"<style.*?</style>", " ", html, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()[:12000]

    if not text:
        return []

    prompt = f"""
You extract product data from ecommerce pages.

Rules:
- You can extract from a single product page or from a category/listing page.
- If it is a category/listing page, return up to {max_products} concrete products that clearly match the query.
- If the page is a blog/news/help/about page, return an empty products list.
- Ignore fake prices like product counts, installment months, ratings, storage sizes, discount percentages, or filter values.
- Currency must be one of: EGP, SAR, AED, USD.
- Availability should be one of: in_stock, out_of_stock, preorder, or null.

URL: {link}
Fallback title: {fallback_title}
Page text:
{text}
""".strip()

    try:
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": GeminiPageExtraction,
                },
            ),
        )
        parsed = response.parsed
    except Exception:
        logger.exception("gemini extraction failed for %s", link)
        return []

    if not parsed or not parsed.products:
        return []

    products = []
    for product in parsed.products[:max_products]:
        if not product.product_name or product.price is None:
            continue
        products.append(
            {
                "title": product.product_name,
                "price": float(product.price),
                "currency": product.currency,
                "availability": product.availability,
            }
        )
    return products


async def find_product_url(listing_url, product_title):
    """
    Tries to find a product URL given a listing page URL and a product title.

    It downloads the listing page HTML, iterates over all `<a href>` anchors,
    skips anchors with fragment, javascript, or mailto hrefs, relative
    hrefs pointing to a different domain, pagination or filter links,
    and keeps only anchors whose path contains a product-like segment
    (`/p/`, `/product/`, `/products/`, `/dp/`).

    It then computes the word overlap between the product title and the
    anchor's text + URL path. If at least one word is shared, the
    anchor's href is returned as the product URL. If no match is found,
    `None` is returned.

    :param listing_url: The URL of the listing page
    :param product_title: The title of the product
    :return: The product URL, or None if no match is found
    """
    try:
        html = await fetch_text(listing_url, timeout=20)
    except aiohttp.ClientError:
        logger.warning("listing fetch failed %s", listing_url)
        return None

    soup = BeautifulSoup(html, "html.parser")
    listing_domain = urlparse(listing_url).netloc
    title_words = set(re.findall(r"[a-z0-9]+", product_title.lower()))

    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        if href.startswith(("#", "javascript:", "mailto:")):
            continue

        full_url = urljoin(listing_url, href)
        parsed = urlparse(full_url)
        path = unquote(parsed.path).lower()

        if parsed.netloc and parsed.netloc != listing_domain:
            continue
        if full_url == listing_url:
            continue
        if any(skip in full_url for skip in ["sort_by=", "items_per_page=", "page-"]):
            continue
        if not any(part in path for part in ["/p/", "/product/", "/products/", "/dp/"]):
            continue

        link_text = anchor.get_text(" ", strip=True).lower()
        candidate_words = set(re.findall(r"[a-z0-9]+", f"{path} {link_text}"))
        if title_words & candidate_words:
            return full_url

    return None


async def resolve_product_urls(source_url, title, extracted_products):
    if not is_listing_like_page(source_url, title):
        return [source_url] * len(extracted_products)

    tasks = [find_product_url(source_url, product["title"]) for product in extracted_products]
    return await asyncio.gather(*tasks)


async def process_shopping_item(item, query, max_stores):
    immersive_url = item.get("serpapi_immersive_product_api")
    if not immersive_url:
        return []

    if "api_key=" not in immersive_url:
        immersive_url = f"{immersive_url}&api_key={SERP_API_KEY}"

    try:
        immersive_response = await fetch_json(immersive_url, timeout=30)
    except aiohttp.ClientError:
        logger.warning("shopping item failed %s", immersive_url)
        return []

    stores = immersive_response.get("product_results", {}).get("stores", [])
    results = []
    for store in stores[:max_stores]:
        source_url = store.get("link")
        price = store.get("extracted_price")
        title = store.get("title") or item.get("title")
        if not source_url or price is None or not title:
            continue

        price_text = store.get("price") or item.get("price")
        results.append(
            ProductResult(
                product_name=title,
                price=float(price),
                currency=get_currency(price_text),
                availability=get_availability(store.get("details_and_offers")),
                source=store.get("name") or item.get("source"),
                source_url=source_url,
                relevance_score=get_score(query, title),
            )
        )
    return results


async def process_organic_item(item, query, max_stores):
    source_url = item.get("link")
    title = item.get("title", "")
    if not source_url or is_obvious_bad_page(source_url, title):
        return []

    extracted_products = await extract_with_gemini(source_url, item.get("title", query), max_stores)
    if not extracted_products:
        if is_listing_like_page(source_url, title):
            return []
        scraped = await scrape_page(source_url, item.get("title", query))
        extracted_products = [scraped] if scraped else []

    resolved_urls = await resolve_product_urls(source_url, title, extracted_products)
    results = []
    for scraped, product_url in zip(extracted_products, resolved_urls):
        if product_url is None:
            if is_listing_like_page(source_url, title):
                continue
            product_url = source_url

        results.append(
            ProductResult(
                product_name=scraped["title"],
                price=scraped["price"],
                currency=scraped["currency"],
                availability=scraped["availability"],
                source=item.get("source"),
                source_url=product_url,
                relevance_score=get_score(query, scraped["title"]),
            )
        )
    return results


def dedupe_results(results):
    deduped = []
    seen_keys = set()
    for result in results:
        result_key = (result.source_url, result.product_name.lower())
        if result_key in seen_keys:
            continue
        seen_keys.add(result_key)
        deduped.append(result)
    return deduped


async def search_products(query, region, max_products=5, max_stores=3):
    if not SERP_API_KEY:
        raise HTTPException(status_code=500, detail="Missing serp_api_key in environment")

    region_data = REGION_CONFIG.get(region)
    if not region_data:
        raise HTTPException(status_code=400, detail=f"Unsupported region: {region}")

    logger.info("search started query=%s region=%s", query, region)

    use_shopping = region_data["gl"] in SUPPORTED_SHOPPING_GLS
    grouped_results = []

    if use_shopping:
        try:
            shopping_response = await fetch_json(
                SERPAPI_URL,
                params={
                    "api_key": SERP_API_KEY,
                    "engine": "google_shopping",
                    "q": f"{query} in {region}",
                    "hl": region_data["hl"],
                    "gl": region_data["gl"],
                    "location": region_data["location"],
                    "google_domain": "google.com",
                },
                timeout=30,
            )
            items = shopping_response.get("shopping_results", [])[:max_products]
            grouped_results = await asyncio.gather(
                *(process_shopping_item(item, query, max_stores) for item in items)
            )
        except aiohttp.ClientError as e:
            logger.warning("shopping search failed (%s), falling back to organic for %s", e, region)
            use_shopping = False

    if not use_shopping:
        try:
            google_response = await fetch_json(
                SERPAPI_URL,
                params={
                    "api_key": SERP_API_KEY,
                    "engine": "google",
                    "q": f"{query} price in {region}",
                    "hl": region_data["hl"],
                    "gl": region_data["gl"],
                    "location": region_data["location"],
                    "google_domain": "google.com",
                },
                timeout=30,
            )
        except aiohttp.ClientError as e:
            logger.error("organic search also failed: %s", e)
            raise HTTPException(status_code=502, detail="Search provider returned an error")
        items = google_response.get("organic_results", [])[:max_products]
        grouped_results = await asyncio.gather(
            *(process_organic_item(item, query, max_stores) for item in items)
        )

    flat_results = [result for group in grouped_results for result in group]
    flat_results = dedupe_results(flat_results)
    flat_results.sort(key=lambda item: (-item.relevance_score, item.price))

    logger.info("search finished query=%s region=%s results=%s", query, region, len(flat_results))
    return flat_results


@app.get("/", response_class=HTMLResponse)
async def home():
    options = "\n".join(
        f'<option value="{region}">{region}</option>'
        for region in REGION_CONFIG.keys()
    )
    return f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Product Search</title>
  <style>
    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      background: #f4f4f4;
      color: #111;
    }}
    .wrap {{
      max-width: 760px;
      margin: 40px auto;
      background: #fff;
      padding: 24px;
      border: 1px solid #ddd;
    }}
    h1 {{
      margin-top: 0;
      font-size: 28px;
    }}
    form {{
      display: grid;
      gap: 12px;
      margin-bottom: 24px;
    }}
    input, select, button {{
      padding: 12px;
      font-size: 16px;
      border: 1px solid #bbb;
    }}
    button {{
      background: #111;
      color: #fff;
      cursor: pointer;
    }}
    .card {{
      border: 1px solid #ddd;
      padding: 16px;
      margin-bottom: 12px;
      background: #fafafa;
    }}
    .muted {{
      color: #666;
      font-size: 14px;
    }}
    a {{
      color: #0a66c2;
      text-decoration: none;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Product Search</h1>
    <form id="search-form">
      <input id="query" placeholder="Search query" value="dell laptop" required>
      <select id="region">{options}</select>
      <button type="submit">Search</button>
    </form>
    <div id="status" class="muted"></div>
    <div id="results"></div>
  </div>
  <script>
    const form = document.getElementById("search-form");
    const status = document.getElementById("status");
    const results = document.getElementById("results");

    form.addEventListener("submit", async (event) => {{
      event.preventDefault();
      results.innerHTML = "";
      status.textContent = "Loading...";

      const payload = {{
        query: document.getElementById("query").value,
        region: document.getElementById("region").value
      }};

      try {{
        const response = await fetch("/search", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify(payload)
        }});

        const data = await response.json();
        if (!response.ok) {{
          status.textContent = data.detail || "Request failed";
          return;
        }}

        status.textContent = `Found ${{data.results.length}} results`;
        results.innerHTML = data.results.map(item => `
          <div class="card">
            <strong>${{item.product_name}}</strong><br>
            <span class="muted">${{item.source || "Unknown source"}}</span><br>
            Price: ${{item.price}} ${{item.currency || ""}}<br>
            Availability: ${{item.availability || "unknown"}}<br>
            Relevance: ${{item.relevance_score}}<br>
            <a href="${{item.source_url}}" target="_blank">Open product</a>
          </div>
        `).join("");
      }} catch (error) {{
        status.textContent = "Something went wrong";
      }}
    }});
  </script>
</body>
</html>
""".strip()


@app.post("/search", response_model=SearchResponse)
async def search_endpoint(payload: SearchRequest):
    logger.info("request received query=%s region=%s", payload.query, payload.region)
    return SearchResponse(results=await search_products(payload.query, payload.region))


async def main():
    results = await search_products("dell laptop", "Saudi", max_products=5, max_stores=3)
    for result in results[:5]:
        print("=" * 20)
        print(result.model_dump())
        print("=" * 20)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    uvicorn.run(app, host="0.0.0.0", port=8000)

