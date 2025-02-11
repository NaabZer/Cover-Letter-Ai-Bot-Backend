import os
import asyncio
import json
from crawl_link_finder import get_about_url_using_llm
from typing import List
from pydantic import BaseModel, Field, TypeAdapter
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from crawl4ai.async_configs import BrowserConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy

# TODO: Break this into several files?


class SkillModel(BaseModel):
    skill: str = Field(..., description="The name of the desired skill")
    occurance: int = Field(..., description="The number of times this skill "
                           "is referenced")


class ValueModel(BaseModel):
    value: str = Field(..., description="The name of the company value")
    occurance: int = Field(..., description="The number of times this value "
                           "is referenced")


class CompanyAboutModel(BaseModel):
    page_title: str = Field(..., description="Title of the web page")
    about: str = Field(..., description="Information about the company")
    soft_skills: List[SkillModel] = Field(..., description="List of "
                                          "soft skills mentioned in "
                                          "the web page")
    values: List[ValueModel] = Field(..., description="List of "
                                     "company values mentioned in "
                                     "the web page")


class CompanyAboutPageModel(BaseModel):
    page_title: str = Field(..., description="Name of webpage the "
                            "information is taken from")
    about: str = Field(..., description="Information about the company "
                       "taken from the webpage")


class CompanyAboutOutputModel(BaseModel):
    pages: List[CompanyAboutPageModel]
    soft_skills: List[SkillModel] = Field(..., description="List of soft "
                                          "skills and their count, as "
                                          "found in the web scrapings")
    values: List[ValueModel] = Field(..., description="List of values of the "
                                     "company, as found in the web scrapings")


class JobPostingModel(BaseModel):
    title: str = Field(...,
                       description="Name of the title of the job position")
    company: str = Field(..., description="Name of the company "
                         "posting the job position")
    about: str = Field(..., description="Information about the company "
                       "posting the job position")
    job_description: str = Field(...,
                                 description="Description, responsiblilities "
                                 "and what will be done in the job position")
    job_requirements: str = Field(..., description="Requirements and nice to "
                                  "haves for the job position")
    list_of_skills: List[SkillModel] = Field(..., description="List of skills "
                                             "mentioned in the job position")
    list_of_values: List[ValueModel] = Field(..., description="List of "
                                             "company values mentioned in "
                                             "the job position")


class JobPostingOutputModel(BaseModel):
    job_posting: JobPostingModel
    about: CompanyAboutOutputModel


async def get_company_info_using_llm(
        urls: list[str], provider: str, lang: str = "english",
        title: str = None, api_token: str = None,
        extra_headers: dict[str, str] = None
):
    print(f"\n--- Extracting Structured Data with {provider} ---")

    browser_config = BrowserConfig(headless=True)

    extra_args = {"temperature": 0, "top_p": 0.9, "max_tokens": 2000}
    if extra_headers:
        extra_args["extra_headers"] = extra_headers

    instruction = ("From the crawled content, extract "
                   "information about a company profile from their website. "
                   "This information will be parsed by an LLM, so extract "
                   "as much information as possible that are relevant "
                   "to get an understanding of the purpose and values of a "
                   "company. "
                   "Do not summarize it, and make sure the information would "
                   "be easily parsed by an LLM model. Put this information in "
                   "the about part of the output schema. "
                   "Also generate a list of soft skills and a list of company "
                   "values that can be infered from the page. "
                   f"If the web page is not in {lang}, "
                   f"translate it into {lang}")

    if title:
        instruction += f"""Make sure the information, skills and values \
                are relevant to a job with the title: {title}"""

    crawler_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        word_count_threshold=1,
        page_timeout=80000,
        extraction_strategy=LLMExtractionStrategy(
            provider=provider,
            api_token=api_token,
            schema=CompanyAboutModel.model_json_schema(),
            instruction=instruction,
            extraction_type="schema",
            extra_args=extra_args,
        ),
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        results = await crawler.arun_many(
            urls=urls,
            config=crawler_config
        )
        return [result.extracted_content for result in results]


async def get_job_info_using_llm(
    url: str, provider: str, lang: str = "english",
    api_token: str = None, extra_headers: dict[str, str] = None
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
            instruction="""From the crawled content, extract \
                    information about a job posting. The job \
                    posting is for some domain within software engineering. \
                    Separate it into: position, company and job description \
                    and information about the company.\
                    Also generate a list of both hard and soft skills that \
                    are mentioned in the page, where the hard skills are \
                    related to software engineering or work in general. \
                    Also generate a list of company values that can be \
                    infered from the about section.""",
            extra_args=extra_args,
        ),
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(
            url=url,
            config=crawler_config
        )
        return result.extracted_content


def get_job_company_info_using_llm(
        job_url: str, provider: str, homepage_url: str = None,
        api_token: str = None, extra_headers: dict[str, str] = None
):
    job_info = asyncio.run(
        get_job_info_using_llm(
            url=job_url,
            provider=provider,
            api_token=api_token
        )
    )

    job_info_obj = TypeAdapter(List[JobPostingModel]).validate_json(
            job_info
            )[0]

    about_info = None
    if homepage_url:
        link_objs = asyncio.run(
            get_about_url_using_llm(
                url=homepage_url,
                provider=provider,
                api_token=api_token
            )
        )
        links = [str(link.url) for link in link_objs]

        about_info_list = asyncio.run(
            get_company_info_using_llm(
                urls=links,
                title=job_info_obj.title,
                provider=provider,
                api_token=api_token
            )
        )

        #  The string [] is 2 characters, therefore len > 2 check
        about_info_obj_list = [TypeAdapter(
            List[CompanyAboutModel]).validate_json(about_info)[0]
                               for about_info in about_info_list
                               if len(about_info) > 2]
        about_list = []
        skill_map = {}
        value_map = {}
        for about in about_info_obj_list:
            about_list.append({
                "page_title": about.page_title,
                "about": about.about
                })
            for skill in about.soft_skills:
                skill_map[skill.skill] = skill_map.setdefault(skill.skill, 0)\
                        + skill.occurance
            skill_map = dict(sorted(value_map.items(),
                                    key=lambda item: item[1])
                             )
            for value in about.values:
                value_map[value.value] = value_map.setdefault(value.value, 0)\
                        + value.occurance
            value_map = dict(sorted(value_map.items(),
                                    key=lambda item: item[1])
                             )
        # TODO: Figure out a way to use pydantic models all the way
        # instead of going from model -> dict -> model
        out_skill_map = [
                SkillModel(**{"skill": item[0], "occurance": item[1]})
                for item in skill_map.items()
                ]
        out_value_map = [
                ValueModel(**{"value": item[0], "occurance": item[1]})
                for item in value_map.items()
                ]
        about_info = {
                "pages": about_list,
                "soft_skills": out_skill_map,
                "values": out_value_map,
                }
        about_out_obj = CompanyAboutOutputModel(**about_info)
    return JobPostingOutputModel(job_posting=job_info_obj, about=about_out_obj)


if __name__ == "__main__":
    # Use ollama with llama3.3
    # asyncio.run(
    #     extract_structured_data_using_llm(
    #         provider="ollama/llama3.3", api_token="no-token"
    #     )
    # )

    job_url = "https://www.tokyodev.com/companies/recursive/jobs/machine-learning-engineer"
    home_url = "https://recursiveai.co.jp/"

    job_info = get_job_company_info_using_llm(
            job_url, homepage_url=home_url,
            provider="gemini/gemini-2.0-flash",
            api_token=os.getenv("GEMENI_API_KEY")
            )
    print(json.dumps(job_info.model_json_schema(), indent=2))
    print()
    print(job_info.model_dump_json(indent=2))
