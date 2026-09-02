from xml.etree import ElementTree

import pytest

from src.ingest.paper_source import _parse_entry, contact_header

pytestmark = pytest.mark.unit

_SAMPLE_ENTRY_XML = """
<entry xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <id>http://arxiv.org/abs/2608.13384v2</id>
  <title>  Structure then Query: A Test   Title  </title>
  <summary>  This is the   abstract text.  </summary>
  <published>2026-08-13T12:00:00Z</published>
  <author><name>Teng Lin</name></author>
  <author><name>Jane Doe</name></author>
  <category term="cs.IR" />
  <category term="cs.DB" />
  <link title="pdf" href="https://arxiv.org/pdf/2608.13384v2" />
</entry>
"""


def test_parse_entry_strips_version_from_id():
    entry = ElementTree.fromstring(_SAMPLE_ENTRY_XML)
    paper = _parse_entry(entry)
    assert paper.arxiv_id == "2608.13384"


def test_parse_entry_collapses_whitespace_in_title_and_abstract():
    entry = ElementTree.fromstring(_SAMPLE_ENTRY_XML)
    paper = _parse_entry(entry)
    assert paper.title == "Structure then Query: A Test Title"
    assert paper.abstract == "This is the abstract text."


def test_parse_entry_collects_all_authors_in_order():
    entry = ElementTree.fromstring(_SAMPLE_ENTRY_XML)
    paper = _parse_entry(entry)
    assert paper.authors == ["Teng Lin", "Jane Doe"]


def test_parse_entry_collects_all_categories():
    entry = ElementTree.fromstring(_SAMPLE_ENTRY_XML)
    paper = _parse_entry(entry)
    assert paper.category == ["cs.IR", "cs.DB"]


def test_parse_entry_parses_published_date():
    entry = ElementTree.fromstring(_SAMPLE_ENTRY_XML)
    paper = _parse_entry(entry)
    assert paper.published_date.isoformat() == "2026-08-13"


def test_parse_entry_extracts_pdf_link_by_title_attribute():
    entry = ElementTree.fromstring(_SAMPLE_ENTRY_XML)
    paper = _parse_entry(entry)
    assert paper.pdf_url == "https://arxiv.org/pdf/2608.13384v2"


def test_parse_entry_handles_missing_published_date():
    xml = """
    <entry xmlns="http://www.w3.org/2005/Atom">
      <id>http://arxiv.org/abs/1234.5678v1</id>
      <title>T</title>
      <summary>S</summary>
    </entry>
    """
    paper = _parse_entry(ElementTree.fromstring(xml))
    assert paper.published_date is None
    assert paper.authors == []
    assert paper.category == []
    assert paper.pdf_url == ""


def test_contact_header_raises_without_email(monkeypatch):
    monkeypatch.delenv("ARXIV_CONTACT_EMAIL", raising=False)
    with pytest.raises(RuntimeError, match="ARXIV_CONTACT_EMAIL"):
        contact_header()


def test_contact_header_includes_email(monkeypatch):
    monkeypatch.setenv("ARXIV_CONTACT_EMAIL", "test@example.com")
    header = contact_header()
    assert "test@example.com" in header
