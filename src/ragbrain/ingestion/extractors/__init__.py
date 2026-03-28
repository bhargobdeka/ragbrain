"""Document extractors.

Imports are lazy to avoid pulling heavy dependencies (PyMuPDF, trafilatura,
slack-sdk) when only one extractor is needed.
"""

__all__ = ["BaseExtractor", "PDFExtractor", "RSSExtractor", "WebExtractor", "SlackExtractor"]


def __getattr__(name: str):
    if name == "BaseExtractor":
        from ragbrain.ingestion.extractors.base import BaseExtractor
        return BaseExtractor
    if name == "PDFExtractor":
        from ragbrain.ingestion.extractors.pdf import PDFExtractor
        return PDFExtractor
    if name == "RSSExtractor":
        from ragbrain.ingestion.extractors.rss import RSSExtractor
        return RSSExtractor
    if name == "WebExtractor":
        from ragbrain.ingestion.extractors.web import WebExtractor
        return WebExtractor
    if name == "SlackExtractor":
        from ragbrain.ingestion.extractors.slack import SlackExtractor
        return SlackExtractor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
