"""
This is a blog post generator flow using the Agno framework.

pip install agno water-ai
"""

import json
from textwrap import dedent
from typing import Dict, List, Optional, Any
import asyncio

from water.core import Flow, create_task
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.newspaper4k import Newspaper4kTools
from agno.utils.log import logger
from agno.workflow import RunResponse
from pydantic import BaseModel, Field
from rich.prompt import Prompt


class NewsArticle(BaseModel):
    title: str = Field(..., description="Title of the article.")
    url: str = Field(..., description="Link to the article.")
    summary: Optional[str] = Field(
        ..., description="Summary of the article if available."
    )


class SearchResults(BaseModel):
    articles: List[NewsArticle]


class ScrapedArticle(BaseModel):
    title: str = Field(..., description="Title of the article.")
    url: str = Field(..., description="Link to the article.")
    summary: Optional[str] = Field(
        ..., description="Summary of the article if available."
    )
    content: Optional[str] = Field(
        ...,
        description="Full article content in markdown format. None if content is unavailable.",
    )

searcher: Agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[DuckDuckGoTools()],
    description=dedent("""\
    You are BlogResearch-X, an elite research assistant specializing in discovering
    high-quality sources for compelling blog content. Your expertise includes:

    - Finding authoritative and trending sources
    - Evaluating content credibility and relevance
    - Identifying diverse perspectives and expert opinions
    - Discovering unique angles and insights
    - Ensuring comprehensive topic coverage\
    """),
    instructions=dedent("""\
    1. Search Strategy 🔍
        - Find 10-15 relevant sources and select the 3 best ones
        - Prioritize recent, authoritative content
        - Look for unique angles and expert insights
    2. Source Evaluation 📊
        - Verify source credibility and expertise
        - Check publication dates for timeliness
        - Assess content depth and uniqueness
    3. Diversity of Perspectives 🌐
        - Include different viewpoints
        - Gather both mainstream and expert opinions
        - Find supporting data and statistics\
    """),
    response_model=SearchResults,
)

# Content Scraper: Extracts and processes article content
article_scraper: Agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[Newspaper4kTools()],
    description=dedent("""\
    You are ContentBot-X, a specialist in extracting and processing digital content
    for blog creation. Your expertise includes:

    - Efficient content extraction
    - Smart formatting and structuring
    - Key information identification
    - Quote and statistic preservation
    - Maintaining source attribution\
    """),
    instructions=dedent("""\
    1. Content Extraction 📑
        - Extract content from the article
        - Preserve important quotes and statistics
        - Maintain proper attribution
        - Handle paywalls gracefully
    2. Content Processing 🔄
        - Format text in clean markdown
        - Preserve key information
        - Structure content logically
    3. Quality Control ✅
        - Verify content relevance
        - Ensure accurate extraction
        - Maintain readability\
    """),
    response_model=ScrapedArticle,
)

# Content Writer Agent: Crafts engaging blog posts from research
writer: Agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    description=dedent("""\
    You are BlogMaster-X, an elite content creator combining journalistic excellence
    with digital marketing expertise. Your strengths include:

    - Crafting viral-worthy headlines
    - Writing engaging introductions
    - Structuring content for digital consumption
    - Incorporating research seamlessly
    - Optimizing for SEO while maintaining quality
    - Creating shareable conclusions\
    """),
    instructions=dedent("""\
    1. Content Strategy 📝
        - Craft attention-grabbing headlines
        - Write compelling introductions
        - Structure content for engagement
        - Include relevant subheadings
    2. Writing Excellence ✍️
        - Balance expertise with accessibility
        - Use clear, engaging language
        - Include relevant examples
        - Incorporate statistics naturally
    3. Source Integration 🔍
        - Cite sources properly
        - Include expert quotes
        - Maintain factual accuracy
    4. Digital Optimization 💻
        - Structure for scanability
        - Include shareable takeaways
        - Optimize for SEO
        - Add engaging subheadings\
    """),
    expected_output=dedent("""\
    # {Viral-Worthy Headline}

    ## Introduction
    {Engaging hook and context}

    ## {Compelling Section 1}
    {Key insights and analysis}
    {Expert quotes and statistics}

    ## {Engaging Section 2}
    {Deeper exploration}
    {Real-world examples}

    ## {Practical Section 3}
    {Actionable insights}
    {Expert recommendations}

    ## Key Takeaways
    - {Shareable insight 1}
    - {Practical takeaway 2}
    - {Notable finding 3}

    ## Sources
    {Properly attributed sources with links}\
    """),
    markdown=True,
)

# --- Define Pydantic Schemas for Task Inputs/Outputs ---
class SearchTaskInput(BaseModel):
    topic: str

class BlogPostOutput(BaseModel):
    title: str
    content: str

class ScrapedArticlesOutput(BaseModel):
    articles: Dict[str, ScrapedArticle]

# --- Define Task Execution Functions ---
def get_search_results(params: Dict[str, Any], context) -> SearchResults:
    """Executes the search logic from get_search_results as a Water task."""
    topic = params["input_data"]["topic"]
    num_attempts: int = 3
    for attempt in range(num_attempts):
        try:
            searcher_response: RunResponse = searcher.run(topic)
            if (
                searcher_response is not None
                and searcher_response.content is not None
                and isinstance(searcher_response.content, SearchResults)
            ):
                article_count = len(searcher_response.content.articles)
                logger.info(
                    f"Found {article_count} articles on attempt {attempt + 1}"
                )
                return searcher_response.content
            else:
                logger.warning(
                    f"Attempt {attempt + 1}/{num_attempts} failed: Invalid response type"
                )
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}/{num_attempts} failed: {str(e)}")

    logger.error(f"Failed to get search results after {num_attempts} attempts")
    return None

def scrape_articles(params: Dict[str, Any], context) -> ScrapedArticlesOutput:
    scraped_articles: Dict[str, ScrapedArticle] = {}
    search_results = params["input_data"]

    # Scrape the articles that are not in the cache
    for article in search_results.articles:
        if article.url in scraped_articles:
            logger.info(f"Found scraped article in cache: {article.url}")
            continue

        article_scraper_response: RunResponse = article_scraper.run(
            article.url
        )
        if (
            article_scraper_response is not None
            and article_scraper_response.content is not None
            and isinstance(article_scraper_response.content, ScrapedArticle)
        ):
            scraped_articles[article_scraper_response.content.url] = (
                article_scraper_response.content
            )
            logger.info(f"Scraped article: {article_scraper_response.content.url}")
    return scraped_articles

def generate_blog_post(params: Dict[str, Any], context) -> BlogPostOutput:
    """Generate a blog post using the Agno writer agent."""
    articles = params["input_data"]
    
    logger.info(f"✍️ Generating blog post")
    writer_input = {
        "articles": [article.model_dump() for article in articles.values()]
    }
    response = writer.run(json.dumps(writer_input, indent=2))
    
    if response and response.content:
        logger.info("✅ Blog post generation complete.")
        return {
            "title": f"Comprehensive Guide",
            "content": response.content
        }
    else:
        logger.error("Failed to generate blog post.")
        return {
            "title": f"Guide",
            "content": "Failed to generate content."
        }

# --- Create the Water Tasks ---
search_task = create_task(
    id="search",
    description="Search for articles using the Agno searcher agent.",
    execute=get_search_results,
    input_schema=SearchTaskInput,
    output_schema=SearchResults
)

scrape_task = create_task(
    id="scrape",
    description="Scrape articles using the Agno article scraper agent.",
    execute=scrape_articles,
    input_schema=SearchResults,
    output_schema=ScrapedArticlesOutput
)

write_task = create_task(
    id="write",
    description="Generate blog post using the Agno writer agent.",
    execute=generate_blog_post,
    input_schema=ScrapedArticlesOutput,
    output_schema=BlogPostOutput
)

# --- Create and Register the Sequential Water Flow ---
blog_flow = Flow(
    id="blog_generation_flow",
    description="Sequential flow to generate a blog post using Agno agents."
)
blog_flow.then(search_task).then(scrape_task).then(write_task).register()

# --- Main Execution Block ---
async def main():
    topic = Prompt.ask(
        "[bold]Enter a blog post topic[/bold] (or press Enter for a random example)\n✨",
    )

    print("\n" + "="*80)
    print(f"🚀 Running Blog Generation for '{topic}' using a Water Sequential Flow 🚀")
    print("="*80 + "\n")
    
    try:
        result = await blog_flow.run({"topic": topic})
        
        print("\n" + "="*80)
        print("🎉 Blog Post Generation Complete! 🎉")
        print("="*80 + "\n")
        print(result.title + "\n")
        print(result.content)
        
    except Exception as e:
        logger.error(f"❌ Flow execution failed: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())