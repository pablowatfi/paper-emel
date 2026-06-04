"""Unit tests for OAI-PMH client — all HTTP mocked with respx, no real calls."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from unittest.mock import patch

import httpx
import pytest
import respx

from arxiv_rag.exceptions import IngestionError
from arxiv_rag.ingestion.oai_client import OAIClient

_OAI_URL = "https://export.arxiv.org/oai2"
_FROM = date(2024, 1, 1)
_UNTIL = date(2024, 1, 15)


async def _noop_sleep(_delay: float) -> None:
    """Replaces asyncio.sleep in retry tests to avoid multi-second waits."""


def _seq(*responses: httpx.Response) -> Callable[[httpx.Request], httpx.Response]:
    """Return a handler that serves responses in order, one per call."""
    queue = list(responses)

    def _handler(_req: httpx.Request) -> httpx.Response:
        return queue.pop(0)

    return _handler


def _record_xml(arxiv_id: str = "2401.12345", token: str = "") -> str:
    token_el = f"<resumptionToken>{token}</resumptionToken>" if token else "<resumptionToken/>"
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
  <ListRecords>
    <record>
      <header>
        <identifier>oai:arXiv.org:{arxiv_id}</identifier>
        <datestamp>2024-01-15</datestamp>
      </header>
      <metadata>
        <arXivRaw xmlns="http://arxiv.org/OAI/arXivRaw/">
          <id>{arxiv_id}</id>
          <title>Test Paper</title>
          <abstract>A test abstract.</abstract>
          <authors>
            <author>
              <keyname>Smith</keyname>
              <forenames>John</forenames>
            </author>
          </authors>
          <categories>cs.LG cs.AI</categories>
          <created>2024-01-15</created>
        </arXivRaw>
      </metadata>
    </record>
    {token_el}
  </ListRecords>
</OAI-PMH>"""


def _no_records_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
  <error code="noRecordsMatch">No records found</error>
</OAI-PMH>"""


class TestHappyPath:
    async def test_single_batch_yields_paper(self, respx_mock: respx.MockRouter) -> None:
        respx_mock.get(_OAI_URL).mock(return_value=httpx.Response(200, text=_record_xml()))

        async with OAIClient(rate_limit_seconds=0.0) as client:
            papers = [p async for p in client.harvest(_FROM, _UNTIL)]

        assert len(papers) == 1
        assert papers[0].arxiv_id == "2401.12345"
        assert papers[0].primary_category == "cs.LG"
        assert papers[0].authors == ["John Smith"]

    async def test_no_records_match_returns_empty(self, respx_mock: respx.MockRouter) -> None:
        respx_mock.get(_OAI_URL).mock(return_value=httpx.Response(200, text=_no_records_xml()))

        async with OAIClient(rate_limit_seconds=0.0) as client:
            papers = [p async for p in client.harvest(_FROM, _UNTIL)]

        assert papers == []

    async def test_deleted_record_is_skipped(self, respx_mock: respx.MockRouter) -> None:
        xml = """<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
  <ListRecords>
    <record>
      <header status="deleted">
        <identifier>oai:arXiv.org:2401.99999</identifier>
        <datestamp>2024-01-15</datestamp>
      </header>
    </record>
    <resumptionToken/>
  </ListRecords>
</OAI-PMH>"""
        respx_mock.get(_OAI_URL).mock(return_value=httpx.Response(200, text=xml))

        async with OAIClient(rate_limit_seconds=0.0) as client:
            papers = [p async for p in client.harvest(_FROM, _UNTIL)]

        assert papers == []


class TestPagination:
    async def test_pagination_via_resumption_token(self, respx_mock: respx.MockRouter) -> None:
        route = respx_mock.get(_OAI_URL)
        route.mock(
            side_effect=_seq(
                httpx.Response(200, text=_record_xml("2401.11111", token="pagetoken")),
                httpx.Response(200, text=_record_xml("2401.22222")),
            )
        )

        async with OAIClient(rate_limit_seconds=0.0) as client:
            papers = [p async for p in client.harvest(_FROM, _UNTIL)]

        assert len(papers) == 2
        assert papers[0].arxiv_id == "2401.11111"
        assert papers[1].arxiv_id == "2401.22222"
        assert route.call_count == 2


class TestRetry:
    async def test_retries_on_503_then_succeeds(self, respx_mock: respx.MockRouter) -> None:
        route = respx_mock.get(_OAI_URL)
        route.mock(
            side_effect=_seq(
                httpx.Response(503),
                httpx.Response(200, text=_record_xml()),
            )
        )

        with patch("asyncio.sleep", _noop_sleep):
            async with OAIClient(rate_limit_seconds=0.0) as client:
                papers = [p async for p in client.harvest(_FROM, _UNTIL)]

        assert len(papers) == 1
        assert route.call_count == 2

    async def test_persistent_503_raises_ingestion_error(
        self, respx_mock: respx.MockRouter
    ) -> None:
        respx_mock.get(_OAI_URL).mock(return_value=httpx.Response(503))

        with patch("asyncio.sleep", _noop_sleep):
            async with OAIClient(rate_limit_seconds=0.0) as client:
                with pytest.raises(IngestionError):
                    async for _ in client.harvest(_FROM, _UNTIL):
                        pass


class TestErrors:
    async def test_malformed_xml_raises_ingestion_error(self, respx_mock: respx.MockRouter) -> None:
        respx_mock.get(_OAI_URL).mock(
            return_value=httpx.Response(200, text="<unclosed>not valid xml <<<")
        )

        async with OAIClient(rate_limit_seconds=0.0) as client:
            with pytest.raises(IngestionError, match="Failed to parse"):
                async for _ in client.harvest(_FROM, _UNTIL):
                    pass

    async def test_oai_error_raises_ingestion_error(self, respx_mock: respx.MockRouter) -> None:
        error_xml = """<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
  <error code="badArgument">Illegal arguments</error>
</OAI-PMH>"""
        respx_mock.get(_OAI_URL).mock(return_value=httpx.Response(200, text=error_xml))

        async with OAIClient(rate_limit_seconds=0.0) as client:
            with pytest.raises(IngestionError, match="badArgument"):
                async for _ in client.harvest(_FROM, _UNTIL):
                    pass
