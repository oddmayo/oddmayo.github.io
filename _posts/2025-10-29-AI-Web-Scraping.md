---
layout: post
title: Leverage local LLMs for lazy scraping
tags: [Scraping, LLM]
categories: Demo
---

[Crawl4AI](https://github.com/unclecode/crawl4ai) is an open source crawling and scraping library that provides many tools for AI ready data extraction. There are many great tutorials out there, but most if not all focus on CSS or XPath extraction strategies that provide a 'result' object in amicable markdown format for LLMs to feed on.

This tutorial focuses instead on using LLMs for obtaining that 'result' object in a lazy way. Why? Because even though you can find the use of this feature in the [Docs](https://docs.crawl4ai.com/), it's hard to find guidance regarding local LLMs. It's easy to just copy and paste your openAI key, but what if you want deeper control of what you are using at relatively zero cost.

# Set up

We are going to use old reliable Ollama, first things first is to install it in our system (consider using vLLM also).

For Linux run on terminal:

``` bash
$ curl -fsSL https://ollama.com/install.sh | sh
```

For Windows and Mac download and install here: [Ollama](https://ollama.com/download)

Verify your installation:

``` bash
$ ollama -v
ollama version is 0.12.7
```

Then we can install crawl4ai:

``` bash
$ pip install -U crawl4ai
```

Run the post-installation setup. I've had little success with this command in two linux systems, but fortunately the next one installs playwright with no problems.

``` bash
$ crawl4ai-setup
```

``` bash
$ python -m playwright install --with-deps chromium
```

That's it for now, we can go to our notebook and check if the installation had no problems.

``` python
# crawl4ai health check
!crawl4ai-doctor
```

```         
[INIT].... → Running Crawl4AI health check... 
[INIT].... → Crawl4AI 0.7.6 
[TEST].... ℹ Testing crawling capabilities... 
[EXPORT].. ℹ Exporting media (PDF/MHTML/screenshot) took 0.38s 
[FETCH]... ↓ https://crawl4ai.com                                                                 
| ✓ | ⏱: 3.91s 
[SCRAPE].. ◆ https://crawl4ai.com                                                              
| ✓ | ⏱: 0.02s 
[COMPLETE] ● https://crawl4ai.com                                                                    
| ✓ | ⏱: 3.93s 
[COMPLETE] ● ✅ Crawling test passed!
```

It's good to check the working status of the library every now and then in case you need to deprecate.

Since we are working on a notebook the next command is neccessary for allowing the execution of nested asyncio event loops (the building block for blazing fast scraping).

``` python
# if running in notebooks
nest_asyncio.apply()
```

# Basic web scraping

This is the example provided in the Docs, briefly explained: we need to define a browser config and a crawler run config for the Crawler. We will focus on single url extraction with the arun() method, but be sure to check out [arun_many()](https://docs.crawl4ai.com/api/arun_many/) (multiple request might need a proxy for large scale scraping).

``` python
async def main():
    browser_conf = BrowserConfig(headless=True)
    run_conf = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS
    )

    async with AsyncWebCrawler(config=browser_conf) as crawler:
        result = await crawler.arun(
            url="https://www.scrapethissite.com/pages/",
            config=run_conf
        )
        print(result.markdown)

if __name__ == "__main__":
    asyncio.run(main())
```

```         
# Web Scraping Sandbox
* * *
###  [Countries of the World: A Simple Example](https://www.scrapethissite.com/pages/simple/)
A single page that lists information about all the countries in the world. Good for those just get started with web scraping. 
* * *
###  [Hockey Teams: Forms, Searching and Pagination](https://www.scrapethissite.com/pages/forms/)
Browse through a database of NHL team stats since 1990. Practice building a scraper that handles common website interface components. 
* * *
###  [Oscar Winning Films: AJAX and Javascript](https://www.scrapethissite.com/pages/ajax-javascript/)
Click through a bunch of great films. Learn how content is added to the page asynchronously with Javascript and how you can scrape it. 
* * *
...
Scraping real websites, you're likely run into a number of common gotchas. Get practice with spoofing headers, handling logins & session cookies, finding CSRF tokens, and other common network errors. 
* * *
Lessons and Videos © Hartley Brody 2023 
```

Pretty standard, if you wanted to extract specific elements you could use any of the [LLM-Free Strategies](https://docs.crawl4ai.com/extraction/no-llm-strategies/). But we are going the opposite route.

# Scraping with local LLM

For this task I'm going to use qwen2.5:3b, a pretty well know model for fast data extraction, you should check the list of [models](https://ollama.com/search) in Ollama and their task rankings in [Hugging Face 🤗](https://huggingface.co/models).

Make sure to pull the model you chose. Don't hoard models, make sure any unused ones with ollama rm model_name.

``` bash
$ ollama pull qwen2.5:3b
```

We can find a simple example of LLM extration in the docs, just make sure to modify the provider, instruction and url:

``` python
class Product(BaseModel):
    name: str
    description: str

async def main():
    # 1. Define the LLM extraction strategy
    llm_strategy = LLMExtractionStrategy(
        llm_config = LLMConfig(provider="ollama/qwen2.5:3b", api_token=None),
        schema=Product.model_json_schema(),
        extraction_type="schema",
        instruction=""" 
        From the crawled content
        extract the titles and the description in JSON format like this:
        {"title": "title name", "description: "description text"}
        """,
        chunk_token_threshold=1000,
        overlap_rate=0.0,
        apply_chunking=False,
        input_format="markdown",   # or "html", "fit_markdown"
        extra_args={"temperature": 0.0, "max_tokens": 500}
    )

    # 2. Build the crawler config
    crawl_config = CrawlerRunConfig(
        extraction_strategy=llm_strategy,
        cache_mode=CacheMode.BYPASS
    )

    # 3. Create a browser config if needed
    browser_cfg = BrowserConfig(
        headless=True,
        text_mode=True,
        light_mode=True
        )

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        # 4. Let's say we want to crawl a single page
        result = await crawler.arun(
            url="https://www.scrapethissite.com/pages/",
            config=crawl_config
        )

        if result.success:
            # 5. The extracted content is presumably JSON
            data = json.loads(result.extracted_content)
            print("Extracted items:", data)

            # 6. Show usage stats
            llm_strategy.show_usage()  # prints token usage
        else:
            print("Error:", result.error_message)
        
        return data 

if __name__ == "__main__":
    asyncio.run(main())
```

<details markdown="1">
  <summary>Click for output</summary>

  * This should now work reliably.
  
  **Bold text** and more.
</details>





## Code highlight

Mode specific code highlighting themes. [Kramdown](https://kramdown.gettalong.org/) which is responsible for the color highlighting may be more limited than your IDE.