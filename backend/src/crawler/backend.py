"""Backend module for scraping different sources (Wikipedia, etc)."""

import tempfile


from anyio import Path

from ..log import get_logger

from typing import (
    Dict,
    Optional
)

from ..utilities import (
    estrattore_dominio,
    validate_supported_domain
)

from .crawler_oop import (
    Crawler,
    WikipediaCrawler,
    EuropeanCouncilCrawler,
    TradingEconomicsCrawler,
    BorsaItalianaCrawler,
)
from ..database.db_manipolation import DatabaseManipulator

DatabaseManipulator = DatabaseManipulator()
logger = get_logger(__name__)


def _get_crawler_for_url(url: str) -> Crawler:
    """
    Restituisce l'istanza crawler associata al dominio dell'URL.

    Args:
        url: URL da associare al crawler corretto (str).

    Returns:
        Istanza del crawler specifico per la sorgente rilevata (Crawler).
    """
    source = estrattore_dominio(url)
    validate_supported_domain(source)

    crawler_map = {
        "en.wikipedia.org": WikipediaCrawler(),
        "www.consilium.europa.eu": EuropeanCouncilCrawler(),
        "tradingeconomics.com": TradingEconomicsCrawler(),
        "www.borsaitaliana.it": BorsaItalianaCrawler(),
    }

    crawler = crawler_map.get(source)
    if crawler is None:
        logger.error(f"Crawler non configurato per il dominio: {source}")
        raise ValueError(f"Crawler non configurato per il dominio: {source}")

    return crawler


async def parse_html_text(url: str, html_text: str) -> Dict[str, str]:
    """
    Esegue il parsing di HTML fornito usando la configurazione del crawler selezionato.

    Args:
        crawler: Crawler da utilizzare per l'estrazione (Crawler).
        url: URL originale del contenuto (str).
        html_text: HTML da parsare (str).

    Returns:
        Dizionario con URL, dominio, titolo, HTML e testo parsato (Dict[str, str]).
    """
    if not html_text.strip():
        logger.error("html_text cannot be empty")
        raise ValueError("html_text cannot be empty")

    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".html", delete=False
        ) as temp_file:
            temp_file.write(html_text)
            temp_path = temp_file.name

        file_url = f"file://{temp_path}"
    except Exception as e:
        logger.error(f"Failed to create temporary file for HTML parsing: {e}")
        raise ValueError(f"Failed to create temporary file for HTML parsing: {e}")

    result = await crawl_url(url, file_url)
    # Cleanup del file temporaneo
    try:
        if temp_path and Path(temp_path).exists():
            Path(temp_path).unlink()
    except Exception as e:
        logger.error(f"Warning: Failed to delete temporary file {temp_path}: {e}")
    if not result:
        logger.error("Parsing failed: no data extracted")
        raise ValueError("Parsing failed: no data extracted")
    return result


async def crawl_url(url: str, file_url: str = "") -> Optional[Dict[str, str]]:
    """
    Esegue il crawling di un URL con il crawler appropriato per sorgente.

    Args:
        url: URL da processare (str).

    Returns:
        Dizionario con i dati estratti oppure None (Optional[Dict[str, str]]).
    """
    crawler = _get_crawler_for_url(url)
    try:
        result = await crawler.crawl(url, file_url)
        return result
    except Exception as e:
        logger.error(f"Crawl failed for {url}: {e}")
        raise
