# parse_exchange_xml.py

import re
from pathlib import Path
from bs4 import BeautifulSoup


def clean(text):
    return re.sub(r"\s+", " ", text).strip()


def parse_exchange_xml(xml_path):

    xml_path = Path(xml_path)

    soup = BeautifulSoup(
        xml_path.read_text(
            encoding="utf-8-sig",
            errors="replace"
        ),
        "html.parser"
    )

    # 제목
    title_tag = soup.find("title")

    title = (
        clean(title_tag.get_text(" ", strip=True))
        if title_tag
        else ""
    )

    lines = [f"# {title}"] if title else []

    # 표만 파싱
    for table in soup.find_all("table"):

        current_section = None

        for tr in table.find_all("tr"):

            cells = [
                clean(cell.get_text(" ", strip=True))
                for cell in tr.find_all(["th", "td"])
            ]

            cells = [
                x for x in cells
                if x
            ]

            if not cells:
                continue

            # 연속 중복 셀 제거
            cleaned = []

            for cell in cells:
                if not cleaned or cell != cleaned[-1]:
                    cleaned.append(cell)

            cells = cleaned

            # 실제 section만 인식
            # 예: "1. 판매ㆍ공급계약 구분"
            # "7.6"은 section으로 인식하지 않음
            section = next(
                (
                    x for x in cells
                    if re.match(
                        r"^\d+\.\s+\S+",
                        x
                    )
                ),
                None
            )

            if section:

                current_section = section

                lines.append("")
                lines.append(
                    f"## {section}"
                )

                cells = [
                    x for x in cells
                    if x != section
                ]

            # key-value
            if len(cells) >= 2:

                key = re.sub(
                    r"^-\s*",
                    "",
                    cells[-2]
                )

                value = cells[-1]

                if key != value:
                    lines.append(
                        f"- {key}: {value}"
                    )

            # 값 하나만 있는 경우
            elif len(cells) == 1:

                value = cells[0]

                if value != current_section:
                    lines.append(
                        f"- {value}"
                    )

    text = "\n".join(lines).strip()

    return {
        "title": title,
        "text": text,
        "char_count": len(text)
    }