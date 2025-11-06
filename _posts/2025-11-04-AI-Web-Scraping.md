---
layout: post
toc: true
title: Leverage local LLMs for lazy scraping
author: oddmayo
feature-img: "assets/img/feature-img/owl.jpg"
thumbnail: "assets/img/thumbnails/feature-img/owl.jpg"
color: rgba(203, 27, 53, 1)
tags: [Scraping, LLM]
categories: scraping
favicon: assets/brain.ico

---

[Crawl4AI](https://github.com/unclecode/crawl4ai) is an open-source crawling and scraping python library that provides a variety of tools for AI-ready data extraction. While there are many excellent tutorials available, most focus on CSS or XPath extraction strategies. Typically, [LLMs](https://www.nvidia.com/en-us/glossary/large-language-models/) are not involved until the result object has been extracted.

However, this tutorial focuses on using LLMs as a lazy extraction strategy. Why? Although you can find information about this feature in the [Docs](https://docs.crawl4ai.com/), it's difficult to find guidance about local LLMs. While it's easy to simply copy and paste your OpenAI key, what if you want more control at zero cost?

**CONTENTS:**

* TOC 
{:toc}

You can download the code from this post here: [oddmayo/crawl4ai-resources](https://github.com/oddmayo/crawl4ai-resources)


**languages:** 

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=white)


**requirements:** \
`crawl4ai==0.7.6` \
`nest_asyncio==1.6.0` \
`pydantic==2.12.3`



# Setup

We're going to use old reliable Ollama. First, we need to install it in our system. Consider using vLLM as well.

For Linux run on terminal:

``` bash
$ curl -fsSL https://ollama.com/install.sh | sh
```

For Windows and Mac OS download and install here: [Ollama](https://ollama.com/download)

Verify your installation:

``` bash
$ ollama -v
ollama version is 0.12.7
```

Then we can install Crawl4AI:

``` bash
$ pip install -U crawl4ai
```

Run the post-installation setup. I've had limited success with this command on two Linux systems, but the next one installs [Playwright](https://playwright.dev/) without any problems.

``` bash
$ crawl4ai-setup
```

``` bash
$ python -m playwright install --with-deps chromium
```

That's it for now. We can go to our notebook, make the imports, and check that the installation was problem-free.

``` python
import nest_asyncio, os, asyncio, json
from pydantic import BaseModel, Field
from typing import Any, List, Optional
from crawl4ai import (
    AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, 
    LLMConfig, LLMExtractionStrategy
)
```

``` python
# crawl4ai health check
!crawl4ai-doctor
```

<details markdown="1">

<summary>Click for output</summary>

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

</details>

It's a good idea to check the working status of the library periodically in case you need to deprecate it.

Since we are working in a notebook, the following command is neccessary to allow the execution of nested [asyncio](https://docs.python.org/3/library/asyncio.html) event loops (the building block for blazing-fast scraping).

``` python
# if running in notebooks
nest_asyncio.apply()
```

# Basic web scraping

This is the example provided in the Docs, briefly explained: We need to define both a browser and a crawler run configuration for the Crawler. We will focus on single URL extraction with the `arun()` method, but be sure to check out [`arun_many()`](https://docs.crawl4ai.com/api/arun_many/) (multiple request may require a proxy for large-scale scraping).

What we see in the website:

<div style="max-width:400px; margin:auto;">
  {% include aligner.html images="posts/crawl4ai/basic.png" column=1 caption="www.scrapethissite.com main content" %}
</div>


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

Execution time: 2.1s

<details markdown="1">

<summary>Click for output</summary>

```         
[INIT].... → Crawl4AI 0.7.6 
[FETCH]... ↓ https://www.scrapethissite.com/pages/                              
| ✓ | ⏱: 1.30s 
[SCRAPE].. ◆ https://www.scrapethissite.com/pages/                              
| ✓ | ⏱: 0.01s 
[COMPLETE] ● https://www.scrapethissite.com/pages/                              
| ✓ | ⏱: 1.31s 
  * [ ![](https://www.scrapethissite.com/static/images/scraper-icon.png) Scrape This Site ](https://www.scrapethissite.com/)
  * [ ](https://www.scrapethissite.com/pages/)
  * [ ](https://www.scrapethissite.com/lessons/)
  * [ ](https://www.scrapethissite.com/faq/)
  * [ Login ](https://www.scrapethissite.com/login/)

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

</details>

Crawl4AI takes care of a lot of things in the background (browser headers, captchas, human behaviour simulation, etc). Make sure to explore all the available parameters to create a more robust scraper.

The output is pretty standard; the HTML is converted into Markdown so any LLM can read it. If you want to extract specific elements you can use any of the [LLM-Free Strategies](https://docs.crawl4ai.com/extraction/no-llm-strategies/), but we are going in the opposite direction.

# Scraping with local LLM

For this task I'm going to use Qwen2.5:3b, a pretty well know model for fast data extraction. You should check the list of [models](https://ollama.com/search) in Ollama and their task rankings in [Hugging Face 🤗](https://huggingface.co/models).

Pull the model:

``` bash
$ ollama pull qwen2.5:3b
```

Don't hoard models. Make sure to delete any you're not using with the command `ollama rm model_name.`

## Simple website

We can find a simple example of LLM extraction in the documentation. Just modify the provider, instruction and URL. We will ask the model to extract the titles descriptions in JSON format. Be careful with chunking; that option is for long websites such as those with infinite scrolling: X (Twitter), Facebook, etc.

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
        {"name": "title name", "description: "description text"}
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

Execution time: 5.9s

<details markdown="1">

<summary>Click for output</summary>

```         
[INIT].... → Crawl4AI 0.7.6 
[FETCH]... ↓ https://www.scrapethissite.com/pages/                              
| ✓ | ⏱: 1.21s 
[SCRAPE].. ◆ https://www.scrapethissite.com/pages/                              
| ✓ | ⏱: 0.00s 
[EXTRACT]. ■ https://www.scrapethissite.com/pages/                              
| ✓ | ⏱: 4.14s 
[COMPLETE] ● https://www.scrapethissite.com/pages/                              
| ✓ | ⏱: 5.36s 
Extracted items: [{'name': 'Countries of the World: A Simple Example', 'description': 'A single page that lists information about all the countries in the world. Good for those just get started with web scraping.'}, {'name': 'Hockey Teams: Forms, Searching and Pagination', 'description': 'Browse through a database of NHL team stats since 1990. Practice building a scraper that handles common website interface components.'}, {'name': 'Oscar Winning Films: AJAX and Javascript', 'description': 'Click through a bunch of great films. Learn how content is added to the page asynchronously with Javascript and how you can scrape it.'}, {'name': 'Turtles All the Way Down: Frames & iFrames', 'description': 'Some older sites might still use frames to break up thier pages. Modern ones might be using iFrames to expose data. Learn about turtles as you scrape content inside frames.'}, {'name': "Advanced Topics: Real World Challenges You'll Encounter", 'description': "Scraping real websites, you're likely run into a number of common gotchas. Get practice with spoofing headers, handling logins & session cookies, finding CSRF tokens, and other common network errors."}]

=== Token Usage Summary ===
Type                   Count
------------------------------
Completion               277
Prompt                   986
Total                  1,263

=== Usage History ===
Request #    Completion       Prompt        Total
------------------------------------------------
1                   277          986        1,263
```

</details>

We can see that we got the same result as before, **but** this time it took a little longer. The upside is that we were able to structure the extracted elements without knowing their HTML tags! Pretty cool, right?

The library also has the option to check the token usage summary per run and history, which is specially important if you're using an API key.

Now that we have an idea of how to use the tool, let's build a proper function to reuse it for multiple websites.

## LLM Scraping Function

To avoid the problems with calling functions from external files, I removed Pydantic. This function provides the following arguments for quick use:

-   url: `str` - your website url.
-   fields: `list` - could be a single element or multiple in plain text format.
-   provider: `str`- your model.

The rest of the arguments have default values, but they can be easily modified. Notice how we are going to use the same prompt for multiple websites.

``` python
async def extract_with_llm(
    url: str,
    fields: List[str],                        # e.g. ["name", "price"] or ["summary"]
    provider: str = "ollama/qwen2.5:3b",      # switch models and providers
    api_token: Optional[str] = None,          # for non-local providers if needed
    instruction: Optional[str] = None,        # site-specific prompt
    max_tokens: int = 500,                    # for more difficult websites
    temperature: float = 0.0,                 # 0 = deterministic
    input_format: str = "markdown",           # or "html"
    apply_chunking: bool = False,             # off unless pages are huge
    chunk_token_threshold: int = 1000,        # only used if chunking
    overlap_rate: float = 0.0,                # only used if chunking
    headless: bool = True,                    # browser settings
    text_mode: bool = True,
    light_mode: bool = True,
    cache_mode: CacheMode = CacheMode.BYPASS  # bypass cache by default
) -> Any:
    """
    Extracts specific fields from a webpage using Crawl4AI's LLM extraction strategy
    """

    # Build a minimal JSON schema manually
    schema = {
        "type": "object",
        "properties": {f: {"type": "string"} for f in fields},
        "required": fields,
    }

    # Default LLM instruction
    if instruction is None:
        example = "{" + ", ".join([f'"{f}": "example {f}"' for f in fields]) + "}"
        instruction = (
            "From the page content, extract these fields in strict JSON with exactly these keys: "
            f"{fields}. Return a JSON object like: {example}"
        )

    # Define the extraction strategy
    llm_strategy = LLMExtractionStrategy(
        llm_config=LLMConfig(provider=provider, api_token=api_token),
        schema=schema,
        extraction_type="schema",
        instruction=instruction,
        chunk_token_threshold=chunk_token_threshold,
        overlap_rate=overlap_rate,
        apply_chunking=apply_chunking,
        input_format=input_format,
        extra_args={
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
    )

    # Configure crawler
    crawl_config = CrawlerRunConfig(
        extraction_strategy=llm_strategy,
        cache_mode=cache_mode,
    )

    browser_cfg = BrowserConfig(
        headless=headless,
        text_mode=text_mode,
        light_mode=light_mode,
    )

    # Run crawl and extract data
    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        result = await crawler.arun(url=url, config=crawl_config)

        if not result.success:
            raise RuntimeError(result.error_message)

        # Return parsed JSON
        try:
            return json.loads(result.extracted_content)
        except Exception:
            # Return raw content if not valid JSON
            return result.extracted_content
```

## Real website

Let's test our function with MicroCenter by asking it to extract the name and price of a product.

<div style="max-width:700px; margin:auto;">
  {% include aligner.html images="posts/crawl4ai/medium.png" column=1 caption="microcenter product main content" %}
</div>

``` python
await extract_with_llm(
    url="https://www.microcenter.com/product/670842/intel-core-i7-14700k-raptor-lake-s-refresh-34ghz-twenty-core-lga-1700-boxed-processor-heatsink-not-included",
    fields=["name","price"],
    provider="ollama/qwen2.5:3b"
)
```

Execution time: 5.1s

<details markdown="1">

<summary>Click for output</summary>

```         
INIT].... → Crawl4AI 0.7.6 
[FETCH]... ↓ 
https://www.microcenter.com/product/670842/intel...e-lga-1700-boxed-processor-he
atsink-not-included  | ✓ | ⏱: 0.94s 
[SCRAPE].. ◆ 
https://www.microcenter.com/product/670842/intel...e-lga-1700-boxed-processor-he
atsink-not-included  | ✓ | ⏱: 0.00s 
[EXTRACT]. ■ 
https://www.microcenter.com/product/670842/intel...e-lga-1700-boxed-processor-he
atsink-not-included  | ✓ | ⏱: 3.51s 
[COMPLETE] ● 
https://www.microcenter.com/product/670842/intel...e-lga-1700-boxed-processor-he
atsink-not-included  | ✓ | ⏱: 4.46s 

[{'name': 'Intel Core i7-14700K Raptor Lake-S Refresh 34GHz Twenty-Core LGA 1700 Boxed Processor Heatsink Not Included',
  'price': '$299.99'}]
```

</details>

It's fast and easy. We should test a harder website.

## Complex website limitations

Amazon is one of the most commonly scraped websites. How would our scraper perform?

<div style="max-width:800px; margin:auto;">
  {% include aligner.html images="posts/crawl4ai/hard.png" column=1 caption="amazon product main content" %}
</div>

``` python
await extract_with_llm(
    url="https://www.amazon.com/Bose-Cancelling-Wireless-Bluetooth-Headphones/dp/B07Q9MJKBV/ref=sr_1_1?sr=8-1",
    fields=["name","price"],
    provider="ollama/qwen2.5:3b"
)
```

Execution time: 9.1s

<details markdown="1">

<summary>Click for output</summary>

```         
[INIT].... → Crawl4AI 0.7.6 
[FETCH]... ↓ 
https://www.amazon.com/Bose-Cancelling-Wireless-Bluetooth-Headphones/dp/B07Q9MJK
BV/ref=sr_1_1?sr=8-1 | ✓ | ⏱: 3.00s 
[SCRAPE].. ◆ 
https://www.amazon.com/Bose-Cancelling-Wireless-Bluetooth-Headphones/dp/B07Q9MJK
BV/ref=sr_1_1?sr=8-1 | ✓ | ⏱: 0.29s 
[EXTRACT]. ■ 
https://www.amazon.com/Bose-Cancelling-Wireless-Bluetooth-Headphones/dp/B07Q9MJK
BV/ref=sr_1_1?sr=8-1 | ✓ | ⏱: 5.22s 
[COMPLETE] ● 
https://www.amazon.com/Bose-Cancelling-Wireless-Bluetooth-Headphones/dp/B07Q9MJK
BV/ref=sr_1_1?sr=8-1 | ✓ | ⏱: 8.52s 

[{'name': 'Bose Cancelling Wireless Bluetooth Headphones', 'price': '$249.00'}]
```

</details>

That took longer than the previous ones. If you look closely, you'll notice that the price is wrong. But if you look even more closely, you'll see that the title is not exactly the same as the one shown in the image. Why are these details inaccurate? Amazon product pages contain multiple prices and recommendations depending on the product.

You could overcome this by fine-tuning to ensure the consistent extraction of the desired elements of the main product. However, for lazy purposes, let's give Amazon the win this time. The other extraction strategies would complete the task without any problems in this case.

# Where LLMs shine

Using an LLM for scraping requires targeting the right websites. For example, target websites that constantly change with lots of text. Let's request a summary of this Harvard master's program.

<div style="max-width:800px; margin:auto;">
  {% include aligner.html images="posts/crawl4ai/harvard.png" column=1 caption="harvard program main content" %}
</div>

``` python
await extract_with_llm(
    url="https://extension.harvard.edu/academics/programs/computer-science-masters-degree-program/#program-overview",
    fields=["program summary"],
    provider="ollama/qwen2.5:3b"
)
```

Execution time: 5.6s

<details markdown="1">

<summary>Click for output</summary>

```         
[INIT].... → Crawl4AI 0.7.6 
[FETCH]... ↓ 
https://extension.harvard.edu/academics/programs...science-masters-degree-progra
m/#program-overview  | ✓ | ⏱: 1.44s 
[SCRAPE].. ◆ 
https://extension.harvard.edu/academics/programs...science-masters-degree-progra
m/#program-overview  | ✓ | ⏱: 0.06s 
[EXTRACT]. ■ 
https://extension.harvard.edu/academics/programs...science-masters-degree-progra
m/#program-overview  | ✓ | ⏱: 3.62s 
[COMPLETE] ● 
https://extension.harvard.edu/academics/programs...science-masters-degree-progra
m/#program-overview  | ✓ | ⏱: 5.12s 

[{'program summary': 'The Computer Science Master’s Degree Program at Harvard Extension School is an advanced degree designed for lifelong learners who want to improve their lives through education. This program offers rigorous academics and innovative teaching capabilities, accessible online, in evenings, or at your own pace.'}]
```

</details>

We took advantage of both extraction and the LLM's summarization capabilities, all while using the same prompt.

## Structure elements

Of course, we could target different sections. It doesn't matter if their location changes as long as what we need is on the website. We can adjust the parameters to achieve consistent results.

``` python
await extract_with_llm(
    url="https://extension.harvard.edu/academics/programs/computer-science-masters-degree-program/#program-overview",
    fields=["title", "featured faculty", "career oppurtunities", "next term"],
    provider="ollama/qwen2.5:3b"
)
```

Execution time: 6.2s

<details markdown="1">

<summary>Click for output</summary>

```         
[INIT].... → Crawl4AI 0.7.6 
[FETCH]... ↓ 
https://extension.harvard.edu/academics/programs...science-masters-degree-progra
m/#program-overview  | ✓ | ⏱: 0.80s 
[SCRAPE].. ◆ 
https://extension.harvard.edu/academics/programs...science-masters-degree-progra
m/#program-overview  | ✓ | ⏱: 0.05s 
[EXTRACT]. ■ 
https://extension.harvard.edu/academics/programs...science-masters-degree-progra
m/#program-overview  | ✓ | ⏱: 4.87s 
[COMPLETE] ● 
https://extension.harvard.edu/academics/programs...science-masters-degree-progra
m/#program-overview  | ✓ | ⏱: 5.73s 
[{'title': 'Computer Science Master’s Degree Program',
  'featured faculty': '[Henry H. Leitner Senior Lecturer on Computer Science, Harvard University](https://extension.harvard.edu/faculty/henry-h-leitner/), [David J. Malan Senior Lecturer on Computer Science, Harvard University](https://extension.harvard.edu/faculty/david-j-malan/)',
  'career oppurtunities': 'Students in our Computer Science Master’s Program build the skills essential to career advancement in computer science, software engineering, and computer and software architecture. Potential job titles include: * Computer Scientist * Software Engineer * Software Developer * Systems Architect * Software Architect',
  'next term': 'January & Spring Course Registration Opens November 6'}]
```

</details>

There we go — consistent, lazy scraping, taking advantage of modern tools.

# Concluding remarks

 I've always considered web scraping to be a boring task, but the tools for the job have improved so much over time that, nowadays, with libraries like this one and AI, extracting web data has become fun and challenging (including the competitor data you crave). Plus, you can do it with far fewer lines of code than with traditional tools like Selenium, which feels refreshing. Be sure to explore the other strategies and capabilities of the Crawl4AI library!
