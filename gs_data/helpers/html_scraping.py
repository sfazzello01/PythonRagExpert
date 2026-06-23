"""Scraping HTML da URL e salvataggio formattato per JSON."""

import asyncio
import webbrowser
import tempfile
from typing import Dict, Optional
from urllib.parse import urlparse
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig


async def scrape_url(url: str) -> Optional[Dict[str, str]]:
    """Scarica HTML completo usando crawl4ai e lo formatta per JSON."""
    try:
        browser_config = BrowserConfig()
        crawler_config = CrawlerRunConfig()

        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(url=url, config=crawler_config)

        domain: str = urlparse(url).netloc

        return {
            "url": url,
            "domain": domain,
            "title": result.metadata.get("title", "No title"),
            "html_text": result.html,
            "gold_text": "",
        }

    except Exception as e:
        print(f"Errore: {e}")
        return None


if __name__ == "__main__":
    url = input("Inserisci l'URL da cui estrarre HTML: ")
    result = asyncio.run(scrape_url(url))
    if result:
        print(f"URL: {result['url']}")
        print(f"Domain: {result['domain']}")
        print(f"Title: {result['title']}")
        print(f"HTML length: {len(result['html_text'])} characters")
        # Salva in JSON
        import json
        with open("gs_data/helpers/scraped_data.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        # Salva l'HTML in un file temporaneo e aprilo nel browser
        import os

        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".html", encoding="utf-8") as tf:
            tf.write(result["html_text"])
            temp_path = tf.name

        webbrowser.open(f"file://{os.path.abspath(temp_path)}")

    
