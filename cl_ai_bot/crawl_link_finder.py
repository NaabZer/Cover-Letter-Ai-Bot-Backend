import os
import asyncio
from typing import Set, List
from pydantic import BaseModel, Field, HttpUrl, TypeAdapter
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from crawl4ai.async_configs import BrowserConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy


class LinkModel(BaseModel):
    link: str = Field(..., description="Name of the link to a page that could \
            contain information about the company")
    url: HttpUrl = Field(..., description="URL of the link to a page that \
            could contain information about the company")

    __hash__ = object.__hash__


class LinksModel(BaseModel):
    list_of_links: Set[LinkModel] = Field(..., description="A list of links \
            on the page that could contain information about the company")


async def get_about_url_using_llm(
        url: str, provider: str, api_token: str = None,
        extra_headers: dict[str, str] = None
):
    print(f"\n--- Extracting Structured Data with {provider} ---")

    browser_config = BrowserConfig(headless=True)

    extra_args = {"temperature": 0, "top_p": 0.9, "max_tokens": 20000}
    if extra_headers:
        extra_args["extra_headers"] = extra_headers

    crawler_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        word_count_threshold=1,
        page_timeout=80000,
        extraction_strategy=LLMExtractionStrategy(
            provider=provider,
            api_token=api_token,
            schema=LinksModel.model_json_schema(),
            extraction_type="schema",
            instruction="From the crawled content, extract \
                    a list of names of buttons and their urls on the page, \
                    where there's a decent chance that the new page would \
                    contain information about the company and most \
                    importantly their values.These shall be unique, so no \
                    duplicate entires in the list. \
                    Make sure the URL's are properly formatted and that \
                    there are no remnants of HTML tags and structure left \
                    in the link. For example, make sure the link does NOT \
                    look like this \
                    https://www.appliedintuition.com/%3C/careers%3E'",
            extra_args=extra_args,
        ),
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(
            url=url,
            config=crawler_config
        )
        result._get_value
        out = []
        links_model = TypeAdapter(List[LinksModel]).validate_json(
                result.extracted_content
                )[0]
        for link in links_model.list_of_links:
            out.append(link)
        return out

if __name__ == "__main__":
    # Use ollama with llama3.3
    # asyncio.run(
    #     extract_structured_data_using_llm(
    #         provider="ollama/llama3.3", api_token="no-token"
    #     )
    # )

    out = asyncio.run(
        get_about_url_using_llm(
            provider="gemini/gemini-2.0-flash",
            api_token=os.getenv("GEMENI_API_KEY")
        )
    )
    print(out)
