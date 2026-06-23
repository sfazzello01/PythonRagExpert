import re
from typing import Any
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy  # type: ignore
from markdownify import markdownify as md  # type: ignore


class MrkdwnFormattingExtractionStrategy(JsonCssExtractionStrategy):
    """
    Estrazione personalizzata ottimizzata che utilizza markdownify e ottimizza
    la pulizia del testo riducendo l'overhead di allocazione.
    """

    def _normalize_line(self, value: str) -> str:
        match = re.match(r"^(\s*)(.*)$", value)
        indent = match.group(1) if match else ""
        content = match.group(2) if match else value
        content = re.sub(r"\s+", " ", content)
        content = re.sub(r"\s+([.,;:!?\)\]\}])", r"\1", content)
        return f"{indent}{content}".rstrip()

    def _cleanup_markdown(self, markdown: str) -> str:
        lines = markdown.replace("\r\n", "\n").splitlines()
        cleaned: list[str] = []
        in_code_fence = False

        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith("```"):
                in_code_fence = not in_code_fence
                cleaned.append(line.rstrip())
                continue
            if in_code_fence:
                cleaned.append(line.rstrip())
                continue
            if not line.strip():
                cleaned.append("")
                continue
            if stripped.startswith("|"):
                cleaned.append(line.rstrip())
                continue
            # Chiamata al metodo di classe
            cleaned.append(self._normalize_line(line))

        return "\n".join(cleaned).strip()

    def _get_element_text(self, element: Any) -> str:
        if element is None:
            return ""

        try:
            markdown = md(str(element), heading_style="ATX")
            # Riutilizza le funzioni della classe senza allocarle di continuo
            return self._cleanup_markdown(markdown)
        except Exception:
            return element.get_text(strip=True)
