from docx import Document
from typing import List, Dict


def docx_parser(file_path: str) -> List[Dict[str, str]]:
    """Segregates DOCX content into a list of dictionaries, each containing one header and its content."""

    document = Document(file_path)

    result: List[Dict[str, str]] = []
    current_header = None
    buffer = []

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue

        is_heading_style = (
            paragraph.style
            and paragraph.style.name
            and paragraph.style.name.startswith("Heading")
        )

        is_visual_header = (
            paragraph.runs
            and all(run.bold for run in paragraph.runs if run.text.strip())
            and len(text) < 120
        )

        if is_heading_style or is_visual_header:
            if current_header is not None:
                result.append(
                    {current_header: "\n".join(buffer).strip()}
                )

            current_header = text
            buffer = []
        else:
            if current_header is not None:
                buffer.append(text)

    if current_header is not None:
        result.append(
            {current_header: "\n".join(buffer).strip()}
        )

    return result
