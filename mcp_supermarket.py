from fastmcp import FastMCP
from pydantic import BaseModel, Field
from typing import Optional
import httpx
import os
from bs4 import BeautifulSoup

mcp = FastMCP(
    name="SupermarketPriceServer",
    instructions="""
You are an Australian grocery price comparison assistant.

Step 1 — Normalise the name firstcategorize the item name FIRST
Check for any typing mistakes for the item namefirst then
Always read grocery://categories to confirm the correct subcategory code
Before calling any tool, map the user's casual item name to subcategory code before building any search payload.

Step 2 — Choose exactly ONE search mode

### Mode A — Broad / Generic Query
Use when the user gives a general item name with NO specific brand.
Examples: "frozen pies", "soft drinks", "yoghurt", "milk", "chips"

Payload rules:
- `search_term`    → can be empty
- `category_codes` → SUBCATEGORY only (e.g. ["frozen-pies-and-pastry"])
- `is_half_price`  → True
- `is_discounted`  → False  (never set both flags True at the same time)
- `brands`         → leave None

### Mode B — Specific Brand / Product Query
Use when the user names a specific brand OR a very specific product.
Examples: "Cadbury Old Gold", "Quilton toilet paper", "Pauls Full Cream Milk"

Payload rules:
- `search_term`    → normalised product name (include brand)
- `category_codes` → SUBCATEGORY for narrowing (optional but recommended)
- `brands`         → brand name(s) extracted from the user query
- `is_discounted`  → True
- `is_half_price`  → False  (never set both flags True at the same time)

## Step 3 — Category vs Subcategory Rule
- Always use the SUBCATEGORY code (e.g. "frozen-pies-and-pastry"), never the
  parent category (e.g. "frozen") — subcategory is more precise.
- Never pass both a parent category AND a subcategory together.
- Only fall back to the parent category if NO matching subcategory exists.

## Step 4 — Fallback when 0 results are returned or you couldn't categorize the subcategory for the item name
If the first call returns total_found = 0:
1. Normalise the `search_term` first and then Retry with `search_term` ONLY and find the `category_codes` based on normalised name for narrowing (optional).
2. Remove `brands`, `is_half_price`, and `is_discounted`.
3. This is a bare keyword search — do not add any other flags except store preference if the user has one.
4.If no results found even after the fallback mode then move to next product and say this product is currently unavailable by "retry_instruction": "Product `item name` currently unavailable in major australian supermarkets"

## Step 5 — Present results
After ANY successful result:
- ALWAYS highlight the `most_discounted` item from the response as the
  top recommendation.
- Show: name, store, current price, was-price, savings, and product URL.
- Use `cupPrice` (price per 100g/unit) when comparing products of different pack sizes for a fair comparison for the best value for money.
"""
)

BASE_URL = "https://api.supermarketsweep.com.au"
API_KEY = os.getenv("SUPERMARKET_SWEEP_API_KEY")


# MODELS 

class PriceInfo(BaseModel):
    sweepId: str
    price: float
    wasPrice: Optional[float] = None
    cupPrice: Optional[float] = None
    cupMeasure: Optional[str] = None
    discountRate: Optional[float] = None
    bestPriceLocation: str
    isInstoreOnly: bool
    isOnlineOnly: bool
    isMultibuy: bool
    isHalfPrice: bool
    isSpecial: bool

    @property
    def is_on_deal(self) -> bool:
        return self.isHalfPrice or self.isMultibuy or self.isSpecial or bool(self.discountRate)

    @property
    def savings(self) -> Optional[float]:
        if self.wasPrice and self.wasPrice > self.price:
            return round(self.wasPrice - self.price, 2)
        return None

    @property
    def effective_discount_pct(self) -> float:
        """Unified discount score regardless of deal flag type."""
        if self.discountRate:
            return self.discountRate
        if self.wasPrice and self.wasPrice > 0:
            return round((self.wasPrice - self.price) / self.wasPrice * 100, 1)
        if self.isHalfPrice:
            return 50.0
        return 0.0


class Product(BaseModel):
    storeCode: str
    code: str
    name: str
    brand: str
    productUrl: str
    packageSize: Optional[str] = None
    categoryCodes: list[str]
    isRestricted: bool
    isAvailable: bool
    price: PriceInfo
    imageUrls: list[str]

    @property
    def store_name(self) -> str:
        stores = {
            "c": "Coles", "w": "Woolworths",
            "cw": "Chemist Warehouse", "p": "Priceline", "a": "Aldi"
        }
        return stores.get(self.storeCode, self.storeCode)

    def to_summary(self) -> dict:
        cup_price_str = None
        if self.price.cupPrice is not None and self.price.cupMeasure:
            cup_price_str = f"${self.price.cupPrice:.2f} per {self.price.cupMeasure.lower()}"

        return {
            "store": self.store_name,
            "name": self.name,
            "brand": self.brand,
            "packageSize": self.packageSize,
            "price": f"${self.price.price:.2f}",
            "cupPrice": cup_price_str,
            "wasPrice": f"${self.price.wasPrice:.2f}" if self.price.wasPrice else None,
            "discountRate": f"{self.price.discountRate}% off" if self.price.discountRate else None,
            "effectiveDiscountPct": self.price.effective_discount_pct,
            "savings": f"${self.price.savings:.2f}" if self.price.savings else None,
            "isHalfPrice": self.price.isHalfPrice,
            "isOnDeal": self.price.is_on_deal,
            "isInstoreOnly": self.price.isInstoreOnly,
            "isOnlineOnly": self.price.isOnlineOnly,
            "isAvailable": self.isAvailable,
            "productUrl": self.productUrl,
            "imageUrl": self.imageUrls[0] if self.imageUrls else None,
        }


class SearchResponse(BaseModel):
    total: int
    products: list[Product]


# API CALL
async def call_search_api(payload: dict) -> SearchResponse:
    """POST to /products/search and return parsed SearchResponse."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            f"{BASE_URL}/products/search",
            headers={
                "x-api-key": API_KEY,
                "Content-Type": "application/json",
            },
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        return SearchResponse(**data)

def _pick_most_discounted(products: list[Product]) -> Optional[dict]:
    """Return the summary of the product with the highest effective discount."""
    if not products:
        return None
    best = sorted(products, key=lambda p: p.price.effective_discount_pct,reverse = True)

    return best


# RESOURCE

@mcp.resource("grocery://categories")
def map_items_to_categories() -> str:
    """Full category taxonomy for Australian supermarket products."""
    return """
Two-level hierarchy: parent categories and subcategories.
Always use SUBCATEGORY codes in tool calls for specificity.
Only use parent category when no matching subcategory exists.

meat-seafood-and-deli - [beef-and-veal, lamb, pork, poultry, seafood, deli, bbq-sausages-and-burgers]
fruit-and-vegetables  - [fruit, vegetables, salad, organic]
bakery                - [in-store-bakery, packaged-bread-and-bakery]
dairy-eggs-and-fridge - [cheese, milk, eggs-butter-and-margarine, cream-custard-and-desserts, yoghurt, dips-and-pate, ready-to-eat-meals]
pantry                - [snacks-and-confectionery, breakfast-and-spreads, baking, oils-and-vinegars, sauces-dressings-and-condiments, herbs-and-spices, canned-food, pasta-rice-and-noodles, tea-and-coffee]
drinks                - [soft-drinks, cordials-and-juices, water, energy-and-sports-drinks, tea, coffee, long-life-milk, flavoured-milk, non-alcoholic-drinks]
frozen                - [frozen-meals, frozen-vegetables, frozen-fruit, ice-cream-and-frozen-desserts, frozen-meat, frozen-seafood, frozen-pies-and-pastry]
household             - [cleaning-goods, laundry, kitchen-and-storage, toilet-paper-tissues-and-paper-towels, pest-control, garden-and-outdoors, clothing-and-accessories, sport-and-fitness, stationery, craft-toys-and-games, electronics]
baby                  - [baby-food-and-baby-formula, nappies-and-wipes, baby-accessories]
pet                   - [dog-and-puppy, cat-and-kitten, birds-fish-and-small-pets]
health-and-beauty     - [cosmetics, dental-care, hair-care, skin-care, vitamins-and-supplements, first-aid-and-medicinal, personal-care, sexual-health, perfume]
beer-wine-and-spirits - [beer, white-wine, red-wine, champagne-and-sparkling, spirits, cask-and-fortified-wine, cider, premixed-drinks]
"""


#TOOL 

@mcp.tool()
async def search_products(
    search_term: str = Field(
        default="",
        description=(
            "product name to search for. "
            "Mode A (broad query):  — rely on category_codes alone. "
            "Mode B (specific brand/product): set to the product name, "
            "including the brand (e.g. 'Cadbury Old Gold 180g'). "
            "Fallback retry: Normalise the search term first and look for category_codes."
        ),
    ),
    store_codes: Optional[list[str]] = Field(
        default=None,
        description=(
            "Optional list of store codes to restrict results to the user's "
            "preferred store. Mapping: Coles → ['c'], Woolworths → ['w'], "
            "Aldi → ['a'], Chemist Warehouse → ['cw'], Priceline → ['p']. "
            "Leave None to search all stores."
        ),
    ),
    category_codes: Optional[list[str]] = Field(
        default=None,
        description=(
            "SUBCATEGORY code(s) from grocery://categories. Always prefer "
            "subcategory over parent category for precision. "
            "Do NOT mix parent + subcategory together. "
            "Required for Mode A. Optional (but recommended) for Mode B. "
            "Omit only on a fallback bare-keyword retry.\n\n"
            "Mode A examples (broad query):\n"
            "  'toilet gel'         → ['cleaning-goods']\n"
            "  'frozen pies'  → ['frozen-pies-and-pastry']\n"
            "  'soft drinks'  → ['soft-drinks']\n\n"
            "Mode B examples (specific brand):\n"
            "  'Cadbury chocolate'    → ['snacks-and-confectionery']\n"
            "  'Quilton toilet paper' → ['toilet-paper-tissues-and-paper-towels']"
        ),
    ),
    brands: Optional[list[str]] = Field(
        default=None,
        description=(
            "Brand name(s) to filter by. Mode B only — extract from the user's "
            "query when they name a specific brand. Leave None for Mode A and "
            "fallback retries.\n\n"
            "Examples:\n"
            "  'Cadbury Old Gold'              → ['Cadbury']\n"
            "  'Quilton or Sorbent toilet paper' → ['Quilton', 'Sorbent']"
        ),
    ),
    is_half_price: bool = Field(
        default=False,
        description=(
            "Mode A ONLY — set True to filter to half-price specials. "
            "Never set True together with is_discounted. "
            "Mode A: is_half_price=True, is_discounted=False. "
            "Mode B: is_half_price=False, is_discounted=True. "
            "Fallback retry: both False."
        ),
    ),
    is_discounted: bool = Field(
        default=False,
        description=(
            "Mode B ONLY — set True to include any discounted product "
            "(not just half-price). Never set True together with is_half_price. "
            "Mode A: is_half_price=True, is_discounted=False. "
            "Mode B: is_half_price=False, is_discounted=True. "
            "Fallback retry: both False."
        ),
    ),
    page_size: int = Field(
        default=20,
        description=(
            "Max products to retrieve from the API for scoring. "
            "Output is automatically capped: 6 products for Mode A, "
            "3 products for Mode B, to conserve context window space."
        ),
    ),
) -> dict:
    """
    Search Australian supermarket products and return the most discounted item.

    ## Mode A — Broad / Generic query (e.g. "milk", "frozen pies", "soft drinks")
    ```
    search_term    = ""
    category_codes = ["milk"]                 # subcategory only
    brands         = None
    is_half_price  = True
    is_discounted  = False
    page_size      = 20
    ```

    ## Mode B — Specific brand/product (e.g. "Cadbury Old Gold", "Pauls Full Cream Milk")
    ```
    search_term    = "Cadbury Old Gold"
    category_codes = ["snacks-and-confectionery"]  # subcategory only
    brands         = ["Cadbury"]
    is_half_price  = False
    is_discounted  = True
    page_size      = 10
    ```

    ## Fallback — 0 results returned by Mode A or B
    ```
    search_term    = "full cream milk"   # normalised item name only
    category_codes = if you can predict the subcategory
    brands         = None
    is_half_price  = False
    is_discounted  = False
    page_size      = 20
    ```

    ## Response
    - `most_discounted`: the single product with the highest effective discount.
      Always present this as the top recommendation.
    - `products`: list capped at 6 (Mode A) or 3 (Mode B / fallback).
    - `search_mode`: indicates which mode was used for transparency.
    """
    # Build payload — only include flags that are True
    payload: dict = {
        "searchTerm": search_term.strip(),
        "sortOrder": "price_asc",
        "pageSize": page_size,
        "pageNumber": 1,
    }

    if category_codes:
        payload["categoryCodes"] = category_codes

    if store_codes:
        payload["storeCodes"] = store_codes

    if brands:
        payload["brands"] = brands

    if is_half_price:
        payload["isHalfPrice"] = True

    if is_discounted:
        payload["isDiscounted"] = True

    result = await call_search_api(payload)

    sorted_products = _pick_most_discounted(result.products)

    # Apply limits based on Mode requested
    # Mode A: 6 products, Mode B: 3 products
    display_limit = 5 if is_half_price else 3
    if result.total == 0:
        return {
            "total_found": 0,
            "search_mode": "no_results",
            "most_discounted": None,
            "products": [],
            "retry_instruction": (
                "No results found. Retry search_products with search_term set "
                "the normalised item name and mutliple category_codes as fallback mode. Clear brands,is_half_price, and is_discounted."
            )
        }
    return {
        "total_found": result.total,
        "search_mode": "broad_half_price" if is_half_price else "specific_brand_discounted",
        "most_discounted": sorted_products[0].to_summary() if sorted_products else None,
        "products": [p.to_summary() for p in sorted_products[:display_limit]] if sorted_products else [],
    }


if __name__ == "__main__":
    mcp.run(transport="sse", host="127.0.0.1", port=8000)
    # mcp.run()
