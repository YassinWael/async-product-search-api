import asyncio

import aiohttp
import pytest
from pydantic import ValidationError

import main
from models import ProductResult, SearchRequest


@pytest.fixture(autouse=True)
def clear_http_cache():
    main.http_cache.clear()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("SAR 2,999.50", (2999.5, "SAR")),
        ("1,250 AED", (1250.0, "AED")),
        ("E£ 899", (899.0, "EGP")),
        ("£1,099.99", (1099.99, "GBP")),
        ("EUR 999", (999.0, "EUR")),
        ("C$ 1,299", (1299.0, "CAD")),
        ("₹74,999", (74999.0, "INR")),
        ("$899", (899.0, "USD")),
        ("QAR 2,499", (2499.0, "QAR")),
        ("125.50 KWD", (125.5, "KWD")),
        ("BHD 299", (299.0, "BHD")),
        ("OMR 450", (450.0, "OMR")),
        ("799 JOD", (799.0, "JOD")),
        ("price unavailable", (None, None)),
    ],
)
def test_extract_price_and_currency(text, expected):
    assert main.extract_price_and_currency(text) == expected


def test_availability_accepts_text_or_list():
    assert main.get_availability("Ready and in stock") == "in_stock"
    assert main.get_availability(["This item is not available"]) == "out_of_stock"
    assert main.get_availability(["Pre-order today"]) == "preorder"


def test_request_model_rejects_blank_query():
    with pytest.raises(ValidationError):
        SearchRequest(query="", region="Saudi Arabia")


def test_product_model_rejects_unsafe_url_scheme():
    with pytest.raises(ValidationError):
        ProductResult(
            product_name="Example",
            price=10,
            currency="USD",
            availability="in_stock",
            source="Example Store",
            source_url="javascript:alert(1)",
            relevance_score=1,
        )


def test_deduplicate_keeps_distinct_sources():
    first = ProductResult(
        product_name="Dell Inspiron 15",
        price=2999,
        currency="SAR",
        availability="in_stock",
        source="Store A",
        source_url="https://store-a.example/product/dell",
        relevance_score=1,
    )
    duplicate = first.model_copy(update={"price": 2899})
    second_source = first.model_copy(
        update={"source": "Store B", "source_url": "https://store-b.example/product/dell"}
    )

    assert main.dedupe_results([first, duplicate, second_source]) == [first, second_source]


def test_listing_page_resolves_product_link(monkeypatch):
    html = """
    <a href="/collections/laptops?page=2">More results</a>
    <a href="/products/dell-inspiron-15">Dell Inspiron 15 laptop</a>
    """

    async def fake_fetch_text(*_args, **_kwargs):
        return html

    monkeypatch.setattr(main, "fetch_text", fake_fetch_text)

    result = asyncio.run(
        main.find_product_url(
            "https://shop.example/collections/laptops",
            "Dell Inspiron 15",
        )
    )

    assert result == "https://shop.example/products/dell-inspiron-15"


def test_shopping_item_keeps_api_key_out_of_url(monkeypatch):
    captured = {}

    async def fake_fetch_json(url, *, params=None, **_kwargs):
        captured.update(url=url, params=params)
        return {
            "product_results": {
                "stores": [
                    {
                        "name": "Example Store",
                        "title": "Dell Inspiron 15",
                        "link": "https://store.example/product/dell",
                        "price": "SAR 2,999",
                        "extracted_price": 2999,
                        "details_and_offers": ["In stock"],
                    }
                ]
            }
        }

    async def allow_test_url(_url):
        return None

    monkeypatch.setattr(main, "fetch_json", fake_fetch_json)
    monkeypatch.setattr(main, "ensure_public_http_url", allow_test_url)
    monkeypatch.setattr(main, "SERP_API_KEY", "private-test-key")

    results = asyncio.run(
        main.process_shopping_item(
            {
                "title": "Dell laptop",
                "serpapi_immersive_product_api": (
                    "https://serpapi.com/immersive?product_id=abc&api_key=provider-key"
                ),
            },
            "dell laptop",
            3,
        )
    )

    assert "private-test-key" not in captured["url"]
    assert "provider-key" not in captured["url"]
    assert captured["params"] == {"product_id": "abc", "api_key": "private-test-key"}
    assert results[0].currency == "SAR"
    assert results[0].availability == "in_stock"


class FakeResponse:
    def __init__(self, status, *, text="", headers=None, url="https://example.com"):
        self.status = status
        self._text = text
        self.headers = headers or {}
        self.url = url

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def read(self):
        return self._text.encode()

    async def text(self):
        return self._text

    def raise_for_status(self):
        if self.status >= 400:
            raise AssertionError(f"unexpected final HTTP status {self.status}")


class FakeSession:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def get(self, *_args, **_kwargs):
        self.calls += 1
        return next(self.responses)


def test_http_retries_transient_status(monkeypatch):
    session = FakeSession([FakeResponse(503), FakeResponse(200, text="ok")])

    async def no_sleep(_seconds):
        return None

    async def allow_test_url(_url):
        return None

    monkeypatch.setattr(main, "http_session", session)
    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    monkeypatch.setattr(main, "ensure_public_http_url", allow_test_url)

    assert asyncio.run(main.fetch_text("https://example.com/product")) == "ok"
    assert session.calls == 2


@pytest.mark.parametrize(
    "url",
    [
        "javascript:alert(1)",
        "http://localhost/private",
        "http://127.0.0.1/private",
        "http://10.0.0.1/private",
    ],
)
def test_rejects_unsafe_outbound_urls(url):
    with pytest.raises(main.UnsafeUrlError):
        asyncio.run(main.ensure_public_http_url(url))


def test_revalidates_redirect_destination(monkeypatch):
    response = FakeResponse(
        302,
        headers={"Location": "http://127.0.0.1/private"},
        url="https://93.184.216.34/start",
    )
    monkeypatch.setattr(main, "http_session", FakeSession([response]))

    with pytest.raises(main.UnsafeUrlError):
        asyncio.run(main.fetch_text("https://93.184.216.34/start"))


def test_rejects_unresolvable_host(monkeypatch):
    async def fail_resolution(_hostname, _port):
        raise OSError("DNS unavailable")

    monkeypatch.setattr(main, "resolve_host_addresses", fail_resolution)

    with pytest.raises(main.UnsafeUrlError):
        asyncio.run(main.ensure_public_http_url("https://missing.example/product"))


def test_cache_stays_bounded(monkeypatch):
    monkeypatch.setattr(main, "MAX_CACHE_ENTRIES", 2)

    main.set_cached_value("first", 1)
    main.set_cached_value("second", 2)
    main.set_cached_value("third", 3)

    assert set(main.http_cache) == {"second", "third"}
    assert main.http_cache["second"][1] == 2
    assert main.http_cache["third"][1] == 3


def test_search_falls_back_to_organic_results(monkeypatch):
    calls = []

    async def fake_fetch_json(_url, *, params=None, **_kwargs):
        calls.append(params["engine"])
        if params["engine"] == "google_shopping":
            raise aiohttp.ClientConnectionError("shopping unavailable")
        return {
            "organic_results": [
                {
                    "title": "Dell Inspiron 15",
                    "link": "https://store.example/product/dell",
                    "source": "Example Store",
                }
            ]
        }

    async def fake_process_organic(item, query, _max_stores):
        return [
            ProductResult(
                product_name=item["title"],
                price=2999,
                currency="SAR",
                availability="in_stock",
                source=item["source"],
                source_url=item["link"],
                relevance_score=main.get_score(query, item["title"]),
            )
        ]

    monkeypatch.setattr(main, "SERP_API_KEY", "test-key")
    monkeypatch.setattr(main, "fetch_json", fake_fetch_json)
    monkeypatch.setattr(main, "process_organic_item", fake_process_organic)

    results = asyncio.run(main.search_products("dell laptop", "Saudi Arabia"))

    assert calls == ["google_shopping", "google"]
    assert len(results) == 1
    assert results[0].product_name == "Dell Inspiron 15"


def test_empty_shopping_results_fall_back_to_organic(monkeypatch):
    calls = []

    async def fake_fetch_json(_url, *, params=None, **_kwargs):
        calls.append(params["engine"])
        if params["engine"] == "google_shopping":
            return {"shopping_results": []}
        return {"organic_results": []}

    monkeypatch.setattr(main, "SERP_API_KEY", "test-key")
    monkeypatch.setattr(main, "fetch_json", fake_fetch_json)

    assert asyncio.run(main.search_products("dell laptop", "Saudi Arabia")) == []
    assert calls == ["google_shopping", "google"]


def test_log_url_redacts_keys_and_tokens():
    url = "https://example.com/search?q=laptop&api_key=secret&access_token=private"
    safe = main.safe_log_url(url)

    assert "secret" not in safe
    assert "private" not in safe
    assert "q=laptop" in safe
