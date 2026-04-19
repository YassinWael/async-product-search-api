from pydantic import BaseModel
from typing import List, Optional


class SearchRequest(BaseModel):
    query:str
    region:str

class ProductResult(BaseModel):
    product_name:str
    price: float
    currency: Optional[str]
    availability: Optional[str]
    source: Optional[str] = None
    source_url:str
    relevance_score:float

class SearchResult(BaseModel):
    title:str
    link:str
    source:str
    price: Optional[str] = None
    extracted_price: Optional[float] = None
    serpapi_immersive_product_api: Optional[str] = None

class SearchResponse(BaseModel):
    results: List[ProductResult]
