import os
import asyncio
from crawl_link_finder import get_about_url_using_llm
from typing import List
from pydantic import BaseModel, Field
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from crawl4ai.async_configs import BrowserConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy


class SkillModel(BaseModel):
    skill: str = Field(..., description="The name of the desired skill")
    occurance: int = Field(..., description="The number of times this skill \
            is referenced")


class ValueModel(BaseModel):
    value: str = Field(..., description="The name of the company value")
    occurance: int = Field(..., description="The number of times this value \
            is referenced")


class CompanyModel(BaseModel):
    about: str = Field(..., description="Information about the company")
    list_of_soft_skills: List[SkillModel] = Field(..., description="List of \
            soft skills mentioned in the web page")
    list_of_values: List[ValueModel] = Field(..., description="List of \
            company values mentioned in the web page")


class JobPostingModel(BaseModel):
    title: str = Field(...,
                       description="Name of the title of the job position")
    company: str = Field(..., description="Name of the company\
                         posting the job position")
    about: str = Field(..., description="Information about the company\
                         posting the job position")
    job_description: str = Field(...,
                                 description="Description of the job position")
    list_of_skills: List[SkillModel] = Field(..., description="List of skills \
            mentioned in the job position")
    list_of_values: List[ValueModel] = Field(..., description="List of \
            company values mentioned in the job position")


async def get_company_info_using_llm(
    urls: list[str],
    provider: str, api_token: str = None, extra_headers: dict[str, str] = None
):
    print(f"\n--- Extracting Structured Data with {provider} ---")

    browser_config = BrowserConfig(headless=True)

    extra_args = {"temperature": 0, "top_p": 0.9, "max_tokens": 2000}
    if extra_headers:
        extra_args["extra_headers"] = extra_headers

    crawler_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        word_count_threshold=1,
        page_timeout=80000,
        extraction_strategy=LLMExtractionStrategy(
            provider=provider,
            api_token=api_token,
            schema=CompanyModel.model_json_schema(),
            extraction_type="schema",
            instruction="From the crawled content, extract \
                    information about a company profile. We want information \
                    that are related to software engineering jobs. \
                    Write a text what the company is about. \
                    Also generate a list of soft skills that are mentioned \
                    in the page. Also generate a list of company values",
            extra_args=extra_args,
        ),
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        results = await crawler.arun_many(
            urls=urls,
            config=crawler_config
        )
        for result in results:
            print(result.extracted_content)


async def get_job_info_using_llm(
    url: str,
    provider: str, api_token: str = None, extra_headers: dict[str, str] = None
):
    print(f"\n--- Extracting Structured Data with {provider} ---")

    browser_config = BrowserConfig(headless=True)

    extra_args = {"temperature": 0, "top_p": 0.9, "max_tokens": 2000}
    if extra_headers:
        extra_args["extra_headers"] = extra_headers

    crawler_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        word_count_threshold=1,
        page_timeout=80000,
        extraction_strategy=LLMExtractionStrategy(
            provider=provider,
            api_token=api_token,
            schema=JobPostingModel.model_json_schema(),
            extraction_type="schema",
            instruction="From the crawled content, extract\
                    information about a job posting. The job posting for some \
                    domain within software engineering. \
                    Separate it into: position, company and job description \
                    and information about the company.\
                    Also generate a list of both hard and soft skills that \
                    are mentioned in the page, related to software \
                    engineering. Also generate a list of company values",
            extra_args=extra_args,
        ),
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(
            url=url,
            config=crawler_config
        )
        print(result.extracted_content)

if __name__ == "__main__":
    # Use ollama with llama3.3
    # asyncio.run(
    #     extract_structured_data_using_llm(
    #         provider="ollama/llama3.3", api_token="no-token"
    #     )
    # )

    job_url = "https://boards.greenhouse.io/appliedintuition/jobs/4330203005?gh_jid=4330203005"

    links = asyncio.run(
        get_about_url_using_llm(
            provider="gemini/gemini-2.0-flash",
            api_token=os.getenv("GEMENI_API_KEY")
        )
    )

    asyncio.run(
        get_job_info_using_llm(
            url=job_url,
            provider="gemini/gemini-2.0-flash",
            api_token=os.getenv("GEMENI_API_KEY")
        )
    )

    asyncio.run(
        get_company_info_using_llm(
            urls=links,
            provider="gemini/gemini-2.0-flash",
            api_token=os.getenv("GEMENI_API_KEY")
        )
    )
