from .backend import (
    crawl_url,
    parse_html_text
)
from .crawler_oop import (
    BorsaItalianaCrawler,
    Crawler,
    EuropeanCouncilCrawler,
    TradingEconomicsCrawler,
    WikipediaCrawler,
)
from .custom_extraction_strategy import MrkdwnFormattingExtractionStrategy

__all__ = [
    "crawl_url",
    "parse_html_text",
    "Crawler",
    "WikipediaCrawler",
    "EuropeanCouncilCrawler",
    "TradingEconomicsCrawler",
    "BorsaItalianaCrawler",
    "MrkdwnFormattingExtractionStrategy",
]
