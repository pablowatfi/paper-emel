"""OAI-PMH client for harvesting arXiv metadata."""

from __future__ import annotations

import asyncio
import xml.etree.ElementTree as ET
from collections.abc import AsyncGenerator
from datetime import UTC, date, datetime

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from arxiv_rag.exceptions import IngestionError
from arxiv_rag.models import Paper

_OAI_BASE = "https://export.arxiv.org/oai2"
_NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "arxiv": "http://arxiv.org/OAI/arXivRaw/",
}
_DEFAULT_RATE_LIMIT = 3.0


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 503
    return isinstance(exc, httpx.ConnectError | httpx.RemoteProtocolError)


def _parse_dt(s: str) -> datetime:
    s = s.strip()
    if not s:
        return datetime(2000, 1, 1, tzinfo=UTC)
    if len(s) == 10:
        s = f"{s}T00:00:00"
    s = s.rstrip("Z")
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


def _parse_paper(record: ET.Element) -> Paper | None:
    """Parse a single OAI-PMH record. Returns None for deleted/unparseable records."""
    header = record.find("oai:header", _NS)
    if header is None:
        return None
    if header.get("status") == "deleted":
        return None

    metadata = record.find("oai:metadata", _NS)
    if metadata is None:
        return None

    arXiv = metadata.find("arxiv:arXivRaw", _NS)
    if arXiv is None:
        return None

    def _text(tag: str) -> str:
        el = arXiv.find(f"arxiv:{tag}", _NS)
        return (el.text or "").strip() if el is not None else ""

    raw_id = _text("id")
    if not raw_id:
        return None

    arxiv_id = raw_id.split("v")[0] if "v" in raw_id else raw_id

    authors_el = arXiv.find("arxiv:authors", _NS)
    authors: list[str] = []
    if authors_el is not None:
        for author_el in authors_el.findall("arxiv:author", _NS):
            keyname = author_el.findtext("arxiv:keyname", namespaces=_NS) or ""
            forenames = author_el.findtext("arxiv:forenames", namespaces=_NS) or ""
            name = f"{forenames} {keyname}".strip() if forenames else keyname.strip()
            if name:
                authors.append(name)

    categories_str = _text("categories")
    categories = [c for c in categories_str.split() if c]
    primary_category = categories[0] if categories else "cs.LG"

    datestamp_el = header.find("oai:datestamp", _NS)
    datestamp = (datestamp_el.text or "").strip() if datestamp_el is not None else ""

    try:
        published_at = _parse_dt(_text("created") or datestamp)
        updated_str = _text("updated")
        updated_at = _parse_dt(updated_str) if updated_str else published_at
    except ValueError as exc:
        raise IngestionError(f"Failed to parse dates for {raw_id}: {exc}") from exc

    try:
        return Paper(
            arxiv_id=arxiv_id,
            title=_text("title") or "(no title)",
            abstract=_text("abstract") or "(no abstract)",
            authors=authors,
            primary_category=primary_category,
            categories=categories,
            published_at=published_at,
            updated_at=updated_at,
            pdf_url=f"https://arxiv.org/pdf/{raw_id}",
            abs_url=f"https://arxiv.org/abs/{raw_id}",
        )
    except Exception as exc:
        raise IngestionError(f"Failed to construct Paper for {raw_id}: {exc}") from exc


class OAIClient:
    """Async OAI-PMH client for harvesting arXiv paper metadata."""

    def __init__(
        self,
        http_client: httpx.AsyncClient | None = None,
        rate_limit_seconds: float = _DEFAULT_RATE_LIMIT,
    ) -> None:
        self._client = http_client or httpx.AsyncClient(timeout=30.0)
        self._owns_client = http_client is None
        self._rate_limit_seconds = rate_limit_seconds

    async def __aenter__(self) -> OAIClient:
        return self

    async def __aexit__(self, *_args: object) -> None:
        if self._owns_client:
            await self._client.aclose()

    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def _fetch_page(self, params: dict[str, str]) -> httpx.Response:
        response = await self._client.get(_OAI_BASE, params=params)
        response.raise_for_status()
        return response

    async def harvest(
        self,
        from_date: date,
        until_date: date,
        set_spec: str = "cs.LG",
    ) -> AsyncGenerator[Paper, None]:
        """Yield Papers from OAI-PMH with resumption-token pagination."""
        params: dict[str, str] = {
            "verb": "ListRecords",
            "metadataPrefix": "arXivRaw",
            "from": from_date.isoformat(),
            "until": until_date.isoformat(),
            "set": set_spec,
        }
        resumption_token: str | None = None

        while True:
            if resumption_token is not None:
                params = {"verb": "ListRecords", "resumptionToken": resumption_token}

            try:
                response = await self._fetch_page(params)
            except httpx.HTTPStatusError as exc:
                raise IngestionError(
                    f"OAI-PMH request failed with status {exc.response.status_code}"
                ) from exc
            except (httpx.ConnectError, httpx.RemoteProtocolError) as exc:
                raise IngestionError(f"OAI-PMH connection failed: {exc}") from exc

            await asyncio.sleep(self._rate_limit_seconds)

            try:
                root = ET.fromstring(response.text)
            except ET.ParseError as exc:
                raise IngestionError(f"Failed to parse OAI-PMH XML: {exc}") from exc

            error_el = root.find("oai:error", _NS)
            if error_el is not None:
                code = error_el.get("code", "unknown")
                if code == "noRecordsMatch":
                    return
                raise IngestionError(f"OAI-PMH error [{code}]: {error_el.text}")

            list_records = root.find("oai:ListRecords", _NS)
            if list_records is None:
                raise IngestionError("OAI-PMH response missing <ListRecords>")

            for record in list_records.findall("oai:record", _NS):
                paper = _parse_paper(record)
                if paper is not None:
                    yield paper

            token_el = list_records.find("oai:resumptionToken", _NS)
            token_text = (token_el.text or "").strip() if token_el is not None else ""
            if not token_text:
                break
            resumption_token = token_text
