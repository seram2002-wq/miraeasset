"""
DART 공시 XML(분기/반기/사업보고서 등)을 섹션 단위로 파싱하는 스크립트

사용법:
    python parse_dart_xml.py <입력.xml> <출력.json>

동작 방식:
    - DART 공시 XML은 목차에 해당하는 항목마다 <TITLE ATOC="Y" AASSOCNOTE="...">를 붙여둔다.
      예) <TITLE ATOC="Y" AASSOCNOTE="L-0-2-3-L1">3. 원재료 및 생산설비</TITLE>
    - 이 TITLE 태그들은 문서 순서대로 형제(sibling) 관계로 나열되어 있으므로,
      "이 TITLE부터 다음 TITLE 직전까지"를 그 섹션의 본문으로 잘라내면
      임베딩 검색 없이도 정확하게 섹션을 분리할 수 있다.
    - AASSOCNOTE 값(예: L-0-2-3-L1)은 그 섹션의 고유 코드로, 나중에
      "II-3 원재료 및 생산설비" 같은 사람이 읽는 라벨과 매핑해 쓸 수 있다.
"""

import sys
import json
from pathlib import Path
from bs4 import BeautifulSoup, NavigableString


def parse_dart_xml(xml_path: str) -> dict:
    with open(xml_path, encoding="utf-8") as f:
        content = f.read()

    soup = BeautifulSoup(content, "lxml-xml")

    # 문서 상단 메타데이터 (기업명, 문서종류 등)
    doc_name = soup.find("DOCUMENT-NAME")
    company_name = soup.find("COMPANY-NAME")

    meta = {
        "rcept_no": Path(xml_path).stem,  # 파일명 = 접수번호인 경우가 많음
        "doc_name": doc_name.get_text(strip=True) if doc_name else None,
        "company_name": company_name.get_text(strip=True) if company_name else None,
    }

    # 목차(TOC)로 표시된 모든 TITLE 태그 = 섹션 헤더 (문서 순서대로 정렬되어 있음)
    titles = soup.find_all("TITLE", ATOC="Y")
    title_ids = set(id(t) for t in titles)  # 다음 섹션 경계 판별용

    sections = []
    for title_tag in titles:
        section_id = title_tag.get("AASSOCNOTE", "")
        title_text = title_tag.get_text(strip=True)

        # 중첩 깊이와 무관하게 "문서에 실제로 나타나는 순서"대로 순회하며
        # 다음 TITLE(ATOC=Y)을 만나기 전까지의 텍스트 노드만 수집
        # (상위 챕터가 하위 섹션 내용까지 삼켜버리는 걸 방지)
        body_parts = []
        for el in title_tag.next_elements:
            if id(el) in title_ids:
                break
            if isinstance(el, NavigableString):
                text = str(el).strip()
                if text:
                    body_parts.append(text)

        body_text = " ".join(body_parts)

        sections.append(
            {
                "section_id": section_id,      # 예: L-0-2-3-L1
                "title": title_text,            # 예: 3. 원재료 및 생산설비
                "text": body_text,
                "char_count": len(body_text),
            }
        )

    return {"meta": meta, "sections": sections}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("사용법: python parse_dart_xml.py <입력.xml> <출력.json>")
        sys.exit(1)

    input_path, output_path = sys.argv[1], sys.argv[2]
    result = parse_dart_xml(input_path)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"기업명: {result['meta']['company_name']}")
    print(f"문서종류: {result['meta']['doc_name']}")
    print(f"섹션 수: {len(result['sections'])}")
    print(f"저장 위치: {output_path}")
