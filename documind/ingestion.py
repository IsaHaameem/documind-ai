import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import PurePosixPath
from uuid import UUID, uuid5
from zipfile import BadZipFile, ZipFile

import pymupdf
from docx import Document as open_docx
from docx.text.paragraph import Paragraph
from langchain_text_splitters import RecursiveCharacterTextSplitter

MAX_EXTRACTED_CHARS = 2_000_000
MAX_PDF_PAGES = 1500
MAX_CHUNKS = 4000
UPLOAD_TYPES = {".pdf", ".docx", ".txt"}


class DocumentError(ValueError):
    pass


@dataclass(frozen=True)
class Passage:
    text: str
    location: str
    page: int | None = None


@dataclass(frozen=True)
class Extraction:
    passages: list[Passage]
    details: dict[str, str]
    page_count: int | None = None


@dataclass(frozen=True)
class Chunk:
    id: UUID
    text: str
    location: str
    page: int | None


def upload_filename(filename: str | None) -> tuple[str, str]:
    name = (filename or "").replace("\\", "/").rsplit("/", 1)[-1].strip()
    if not name or len(name) > 255 or any(not character.isprintable() for character in name):
        raise DocumentError("Invalid filename.")
    extension = PurePosixPath(name).suffix.lower()
    if extension not in UPLOAD_TYPES:
        raise DocumentError("Supported formats are PDF, DOCX and TXT.")
    return name, extension


def check_file_header(content: bytes, extension: str) -> None:
    if not content:
        raise DocumentError("The file is empty.")
    if extension == ".pdf" and not content.startswith(b"%PDF-"):
        raise DocumentError("The file does not have a valid PDF header.")
    if extension == ".docx" and not content.startswith(b"PK\x03\x04"):
        raise DocumentError("The file does not have a valid DOCX header.")


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"[^\S\n]+", " ", text.replace("\x00", "")).strip(" ")


def extract(content: bytes, extension: str) -> Extraction:
    check_file_header(content, extension)
    try:
        match extension:
            case ".pdf":
                result = extract_pdf(content)
            case ".docx":
                result = extract_docx(content)
            case ".txt":
                encoding = "utf-16" if content.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
                text = content.decode(encoding)
                if any(not (character.isprintable() or character in "\t\n\r") for character in text):
                    raise DocumentError("The TXT document contains unsupported control characters.")
                result = Extraction([Passage(clean_text(text), "Text")], {"encoding": encoding})
            case _:
                raise DocumentError("Unsupported document format.")
    except DocumentError:
        raise
    except UnicodeError:
        raise DocumentError("TXT must use UTF-8 or UTF-16 with a byte-order mark.") from None
    except Exception:
        raise DocumentError("The document is damaged or could not be parsed.") from None

    if not any(passage.text.strip() for passage in result.passages):
        raise DocumentError("No extractable text was found. Scanned PDFs require OCR.")
    if sum(len(passage.text) for passage in result.passages) > MAX_EXTRACTED_CHARS:
        raise DocumentError("Extracted text exceeds the processing limit.")
    return result


def extract_pdf(content: bytes) -> Extraction:
    with pymupdf.open(stream=content, filetype="pdf") as pdf:
        if pdf.needs_pass:
            raise DocumentError("Password-protected PDFs are unsupported.")
        if len(pdf) > MAX_PDF_PAGES:
            raise DocumentError(f"PDF exceeds the {MAX_PDF_PAGES}-page processing limit.")
        passages = []
        extracted_chars = 0
        for page in pdf:
            text = clean_text(page.get_text(sort=True))
            if text.strip():
                extracted_chars += len(text)
                if extracted_chars > MAX_EXTRACTED_CHARS:
                    raise DocumentError("Extracted text exceeds the processing limit.")
                number = page.number + 1
                passages.append(Passage(text, f"Page {number}", number))
        metadata = pdf.metadata or {}
        details = {key: metadata[key][:500] for key in ("title", "author") if metadata.get(key)}
        return Extraction(passages, details, len(pdf))


def extract_docx(content: bytes) -> Extraction:
    try:
        with ZipFile(BytesIO(content)) as archive:
            entries = archive.infolist()
            if len(entries) > 2000 or sum(entry.file_size for entry in entries) > 64 * 1024**2:
                raise DocumentError("Expanded DOCX exceeds the processing limit.")
            names = set(archive.namelist())
            if not {"[Content_Types].xml", "word/document.xml"}.issubset(names):
                raise DocumentError("The file is not a valid DOCX document.")
    except BadZipFile:
        raise DocumentError("The DOCX archive is damaged.") from None
    document = open_docx(BytesIO(content))
    passages = []
    for number, block in enumerate(document.iter_inner_content(), 1):
        if isinstance(block, Paragraph):
            text = clean_text(block.text)
            if text.strip():
                passages.append(Passage(text, f"Paragraph block {number}"))
        else:
            for row_number, row in enumerate(block.rows, 1):
                text = clean_text(" | ".join(cell.text for cell in row.cells))
                if text.strip(" |\n"):
                    passages.append(Passage(text, f"Table block {number}, row {row_number}"))
    details = {key: value[:500] for key in ("title", "author")
               if (value := getattr(document.core_properties, key))}
    return Extraction(passages, details)


def split_document(document_id: UUID, extraction: Extraction, size: int, overlap: int) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap, add_start_index=True)
    chunks = []
    for section, passage in enumerate(extraction.passages):
        for part, split in enumerate(splitter.create_documents([passage.text])):
            text = split.page_content
            location = passage.location
            if location == "Text":
                start = split.metadata["start_index"]
                first = passage.text.count("\n", 0, start) + 1
                last = first + text.count("\n")
                location = f"Line {first}" if first == last else f"Lines {first}–{last}"
            chunks.append(Chunk(uuid5(document_id, f"{section}:{part}"), text, location, passage.page))
            if len(chunks) > MAX_CHUNKS:
                raise DocumentError("The document produces too many chunks.")
    return chunks
