import json
from abc import ABC, abstractmethod
from typing import Any
from pathlib import Path
from crawl4ai import (  # type: ignore
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
)

from config import CRAWLED_DIR

from ..utilities import (
    to_string,
    remove_citations,
    file_name_sanitize,
    url_exists_in_gs,
    get_saved_gs_html_path,
    save_gs_html,
)
from .custom_extraction_strategy import MrkdwnFormattingExtractionStrategy
from ..log import get_logger

logger = get_logger(__name__)


class Crawler(ABC):
    """Base class for all web crawlers."""

    # Override in subclasses for domain-specific configurations
    DOMAIN: str
    SCHEMA: dict[str, Any]
    WAIT_FOR: str | None = None
    PAGE_TIMEOUT: int | None = None

    def __init__(self):
        """
        Inizializza la configurazione browser comune ai crawler.

        Args:
            Nessuno.

        Returns:
            Nessun valore di ritorno (None).
        """
        self.browser_config = BrowserConfig(headless=True)

    @property
    def domain(self) -> str:
        """
        Restituisce il dominio associato al crawler.

        Args:
            Nessuno.

        Returns:
            Dominio della sorgente gestita (str).
        """
        return self.DOMAIN

    @abstractmethod
    def process_content(self, item: dict[str, Any]) -> tuple[str, str]:
        """
        Elabora il contenuto estratto e produce titolo e testo parsato.

        Args:
            item: Dati estratti dal crawler (dict[str, Any]).

        Returns:
            Coppia con titolo e testo parsato (tuple[str, str]).
        """
        pass

    def _create_extraction_strategy(
        self,
    ) -> MrkdwnFormattingExtractionStrategy:
        """
        Crea la strategia di estrazione JSON/CSS per il crawler corrente.

        Args:
            Nessuno.

        Returns:
            Strategia di estrazione configurata (MrkdwnFormattingExtractionStrategy).
        """
        return MrkdwnFormattingExtractionStrategy(self.SCHEMA, verbose=True)

    def _extract_title_and_content(
        self,
        item: dict[str, Any],
        content_key: str = "content",
    ) -> tuple[str, str]:
        """
        Estrae titolo e contenuto principale da un item strutturato.

        Args:
            item: Item estratto da cui leggere i campi (dict[str, Any]).
            content_key: Nome del campo che contiene il contenuto principale (str).

        Returns:
            Coppia titolo/contenuto testuale (tuple[str, str]).
        """
        title = to_string(item.get("title", "No title"))
        content = to_string(item.get(content_key, []))
        return title, content

    def get_crawler_config(self) -> CrawlerRunConfig:
        """
        Costruisce la configurazione di esecuzione del crawler.

        Args:
            Nessuno.

        Returns:
            Configurazione pronta per arun (CrawlerRunConfig).
        """
        config_kwargs = {"extraction_strategy": self._create_extraction_strategy()}

        if self.WAIT_FOR:
            config_kwargs["wait_for"] = self.WAIT_FOR
        if self.PAGE_TIMEOUT:
            config_kwargs["page_timeout"] = self.PAGE_TIMEOUT

        return CrawlerRunConfig(**config_kwargs)

    async def crawl(self, url: str, file_url: str = "") -> dict[str, str]:
        """
        Esegue il crawling di un URL e restituisce i dati estratti.

        Args:
            url: URL da processare (str).
            file_url: URL del file HTML da parsare (str).

        Returns:
            Dizionario con URL, dominio, titolo, HTML e testo parsato (dict[str, str]).
        """

        # Controlla se l'URL è nel gold standard
        crwl_url = url
        if url_exists_in_gs(url):
            existing_file_path = get_saved_gs_html_path(url)
            if existing_file_path:
                logger.info(
                    "✓ URL trovato nel gold standard, file locale gia presente..."
                )
                crwl_url = f"file://{existing_file_path}"
                logger.info(f"✓ Usando: {crwl_url}")
            else:
                logger.info("✓ URL trovato nel gold standard, salvo file locale...")
                file_path = save_gs_html(url)
                if file_path:
                    # Converti il path a un URL file://
                    crwl_url = f"file://{file_path}"
                    logger.info(f"✓ Usando: {crwl_url}")
        # se viene passato un file_url, usalo al posto dell'URL originale (utile per il parsing di HTML salvati localmente)
        elif file_url:
            crwl_url = file_url
            logger.info(f"✓ Usando file_url: {crwl_url}")

        async with AsyncWebCrawler(config=self.browser_config) as crawler:
            result = await crawler.arun(url=crwl_url, config=self.get_crawler_config())
            if not result.success:
                raise Exception(f"Errore: {result.error_message}")

            data = json.loads(result.extracted_content)
            item = data[0] if isinstance(data, list) else data

            title, parsed_text = self.process_content(item)
            html_text = self._extract_html(result, url)

            output = {
                "url": url,
                "domain": self.domain,
                "title": title,
                "html_text": html_text,
                "parsed_text": parsed_text,
            }

            self._save_to_file(title, output)
            return output

    def _extract_html(self, result: Any, url: str) -> str:
        """
        Estrae il campo HTML dal risultato del crawler.

        Args:
            result: Oggetto risultato restituito dal crawler (Any).
            url: URL processato, mantenuto per coerenza di firma (str).

        Returns:
            HTML estratto oppure stringa vuota in caso di errore (str).
        """
        try:
            if hasattr(result, "html"):
                return result.html
        except Exception:
            return ""

    def _save_to_file(self, title: str, output: dict[str, str]) -> None:
        """
        Salva su file JSON il risultato del crawling.

        Args:
            title: Titolo usato per costruire il nome file (str).
            output: Payload da serializzare su disco (dict[str, str]).

        Returns:
            Nessun valore di ritorno (None).
        """
        safe_title = file_name_sanitize(title)
        filename = CRAWLED_DIR / f"crawled_{safe_title}.json"

        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        logger.info(f"✓ Salvato: {filename}")


class WikipediaCrawler(Crawler):  # funziona bene!
    """Crawler for Wikipedia articles."""

    DOMAIN = "en.wikipedia.org"

    SCHEMA: dict[str, Any] = {
        "name": "Articolo Wikipedia",
        "baseSelector": "body",
        "fields": [
            {"name": "title", "selector": ".mw-page-title-main", "type": "text"},
            {
                "name": "paragraphs",
                "selector": "#mw-content-text p, #mw-content-text h2, #mw-content-text h3, \
                    #mw-content-text li:not(.sidebar li, .infobox li), .wikitable",
                "type": "list",
                "fields": [{"name": "text", "type": "text"}],
            },
        ],
    }

    def process_content(self, item: dict[str, Any]) -> tuple[str, str]:
        """
        Elabora il contenuto estratto da una pagina Wikipedia.

        Args:
            item: Dati estratti dal parser CSS/JSON (dict[str, Any]).

        Returns:
            Titolo e testo ripulito della pagina (tuple[str, str]).
        """
        title, paragraphs = self._extract_title_and_content(
            item, content_key="paragraphs"
        )

        paragraphs = remove_citations(paragraphs)

        if "See also" in paragraphs:
            paragraphs = paragraphs.split("See also")[0].strip()

        if "References" in paragraphs:
            paragraphs = paragraphs.split("References")[0].strip()

        return title, paragraphs


class EuropeanCouncilCrawler(Crawler):
    """Crawler for European Council press releases."""

    DOMAIN = "www.consilium.europa.eu"

    SCHEMA: dict[str, Any] = {
        "name": "European Council Press Release",
        "baseSelector": "body",
        "fields": [
            {"name": "title", "selector": "h1", "type": "text"},
            {
                "name": "content",
                "selector": ".gsc-p-leading p:not(.gsc-card__body), \
                    .gsc-bge-grid__area p:not(.gsc-card__body), \
                        .gsc-bge-grid__area strong:not(.gsc-card__body), \
                            .gsc-bge-grid__area h3:not(.gsc-card__body), \
                                .gsc-bge-grid__area h2:not(.gsc-card__body), \
                                    .gsc-bge-grid__area #text:not(.gsc-card__body),\
                                        .Theme-Layer-BodyText p, \
                                        .Theme-Layer-BodyText h2, \
                                        .Theme-Layer-BodyText h3, \
                                        .Theme-Layer-BodyText ul, \
                                        .Theme-Layer-BodyText strong,\
                                            .Theme-Layer-TextBlock-Inner.DisplayContainer--heightBasis, \
                                                .gsc-blockquote__quote, \
                                                    .gsc-table__container",
                "type": "list",
                "fields": [{"name": "text", "type": "text"}],
            },
        ],
    }

    def process_content(self, item: dict[str, Any]) -> tuple[str, str]:
        """
        Elabora il contenuto estratto da comunicati del Consiglio Europeo.

        Args:
            item: Dati estratti dal parser CSS/JSON (dict[str, Any]).

        Returns:
            Titolo e testo normalizzato del contenuto (tuple[str, str]).
        """
        title, content = self._extract_title_and_content(item)

        # Elimina tutto dopo "See also"
        if "See also" in content:
            content = content.split("See also")[0].strip()
        if "About this site" in content:
            content = content.split("About this site")[0].strip()
        if "Photos/credits" in content:
            content = content.split("Photos/credits")[0].strip()
    
        return title, content


class TradingEconomicsCrawler(Crawler):
    """
    Crawler for Trading Economics economic indicators.
    Il crawl funziona sulla sezione indicatori economici
    """

    DOMAIN = "tradingeconomics.com"

    SCHEMA: dict[str, Any] = {
        "name": "Trading Economics Indicator",
        "baseSelector": "body",
        "fields": [
            {
                "name": "title",
                "selector": ".page-title",
                "type": "text",
            },
            {
                "name": "content",
                "selector": ".tab-content h2, .col-xl-8.col-lg-8.card-header, .table.table-hover,\
                    .col-xl-8.col-lg-8.card-body:not(:has(> a)), #ctl00_ContentPlaceHolder1_ctl00_ctl03_Panel1,\
                        .table.table-hover.sortable-theme-minimal.table-heatmap.table-striped, \
                            #ctl00_ContentPlaceHolder1_ctl00_PanelDescription",
                "type": "list",
                "fields": [{"name": "text", "type": "text"}],
            },
        ],
    }

    def process_content(self, item: dict[str, Any]) -> tuple[str, str]:
        """
        Elabora il contenuto estratto da una pagina di Trading Economics.

        Args:
            item: Dati estratti dal parser CSS/JSON (dict[str, Any]).

        Returns:
            Titolo e testo aggregato del contenuto (tuple[str, str]).
        """

        #taglia quello che viene dopo "News Stream"
        content = to_string(item.get("content", []))
        if "News Stream" in content:
            content = content.split("News Stream")[0].strip()

        title, content = self._extract_title_and_content(item)

        return title, content


class BorsaItalianaCrawler(Crawler):
    """
    Crawler for Borsa Italiana stock information.
    Il crawl funziona sulla sezione indici/indici_in_continua e sulla sezione notizie
    """

    DOMAIN = "www.borsaitaliana.it"

    SCHEMA: dict[str, Any] = {
        "name": "Borsa Italiana Stock Info",
        "baseSelector": "body",
        "fields": [
            {"name": "title", "selector": "h1", "type": "text"},
            {
                "name": "content",
                "selector": ".onlycontent, .l-grid__cell, .summary-value",
                "type": "list",
                "fields": [{"name": "text", "type": "text"}],
            },
        ],
    }

    def process_content(self, item: dict[str, Any]) -> tuple[str, str]:
        """
        Elabora il contenuto estratto da una pagina di Borsa Italiana.

        Args:
            item: Dati estratti dal parser CSS/JSON (dict[str, Any]).

        Returns:
            Titolo e testo aggregato del contenuto (tuple[str, str]).
        """
        title, content = self._extract_title_and_content(item)

        return title, content
