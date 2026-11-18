from io import BytesIO
from uuid import uuid4
from zipfile import ZipFile

import pymupdf
import pytest
from docx import Document

from documind.ingestion import DocumentError, extract, split_document, upload_filename


def test_pdf_citations_keep_original_page_numbers():
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((50, 50), "Refund policy")
        pdf.new_page()
        pdf.new_page().insert_text((50, 50), "Warranty policy")
        data = pdf.tobytes()
    extraction = extract(data, ".pdf")
    document_id = uuid4()
    chunks = split_document(document_id, extraction, 100, 10)
    assert extraction.page_count == 3
    assert [chunk.page for chunk in chunks] == [1, 3]
    assert chunks == split_document(document_id, extraction, 100, 10)
    assert chunks[0].id != split_document(uuid4(), extraction, 100, 10)[0].id


def test_docx_preserves_paragraph_and_table_order():
    docx = Document()
    docx.add_paragraph("Policy introduction")
    table = docx.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Retained for"
    table.cell(0, 1).text = "90 days"
    docx.add_paragraph("Policy conclusion")
    buffer = BytesIO()
    docx.save(buffer)
    actual = extract(buffer.getvalue(), ".docx")
    assert [(passage.location, passage.text) for passage in actual.passages] == [
        ("Paragraph block 1", "Policy introduction"),
        ("Table block 2, row 1", "Retained for | 90 days"),
        ("Paragraph block 3", "Policy conclusion"),
    ]


def test_txt_preserves_line_location_and_supports_windows_line_endings():
    rows = ["A" * 95, "B" * 95, "C" * 95]
    result = extract("\r\n".join(rows).encode("utf-16"), ".txt")
    chunks = split_document(uuid4(), result, 100, 5)
    assert [chunk.location for chunk in chunks] == ["Line 1", "Line 2", "Line 3"]
    assert [chunk.text for chunk in chunks] == rows


@pytest.mark.parametrize(("content", "extension", "reason"), [
    (b"", ".txt", "empty"),
    (b" \r\n ", ".txt", "No extractable text"),
    (b"\x91\xd5", ".txt", "UTF-8"),
    (b"abc\x00", ".txt", "control characters"),
    (b"plain text", ".pdf", "PDF header"),
    (b"PK\x03\x04garbage", ".docx", "damaged"),
])
def test_invalid_files(content, extension, reason):
    with pytest.raises(DocumentError, match=reason):
        extract(content, extension)


def test_wrong_zip_structure_is_rejected():
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("arbitrary.txt", "not an office document")
    with pytest.raises(DocumentError, match="valid DOCX"):
        extract(buffer.getvalue(), ".docx")


def test_upload_name_ignores_path_and_rejects_unsupported_extension():
    assert upload_filename(r"C:\other\Report.PDF") == ("Report.PDF", ".pdf")
    with pytest.raises(DocumentError, match="Supported"):
        upload_filename("report.pdf.exe")
