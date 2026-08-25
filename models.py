from pydantic import BaseModel, Field, HttpUrl


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=200)
    region: str = Field(min_length=2, max_length=50)


class ProductResult(BaseModel):
    product_name: str
    price: float = Field(ge=0)
    currency: str | None
    availability: str | None
    source: str | None = None
    source_url: HttpUrl
    relevance_score: float = Field(ge=0, le=1)


class SearchResponse(BaseModel):
    results: list[ProductResult]
