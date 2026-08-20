# -*- coding: utf-8 -*-
"""
DART 거래소공시 XML -> Markdown -> Chunk (공통 스키마) 파이프라인
================================================================
담당 서식: exchange (거래소공시 - 단일판매공급계약, 신규시설투자, 투자판단 등)

사용:
    python3 dart_exchange_chunker.py <xml_path> <manifest.jsonl> [--out-dir DIR]

산출물 (out-dir 기준):
    {rcept_no}.md          - 문서 전체를 재구성한 마크다운 (검수/디버깅용)
    {rcept_no}.chunks.jsonl - 공통 스키마로 청킹된 결과 (임베딩 입력용)
"""

import json
import re
import sys
import argparse
from pathlib import Path
from bs4 import BeautifulSoup


def clean_txt(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def extract_exchange_blocks(xml_path: Path):
    """
    거래소 공시 XML(HTML/XForms)을 파싱하여 제목, 회사명, 섹션 블록 목록을 반환
    """
    content = xml_path.read_text(encoding="utf-8-sig", errors="replace")
    soup = BeautifulSoup(content, "html.parser")

    title_tag = soup.find("title") or soup.find(class_="xforms_title")
    doc_title = clean_txt(title_tag.get_text(" ", strip=True)) if title_tag else ""

    # 제목에서 회사명 / 보고서명 분리 시도
    company_name = ""
    report_name = doc_title
    if "/" in doc_title:
        parts = [p.strip() for p in doc_title.split("/") if p.strip()]
        if len(parts) >= 2:
            company_name = parts[0]
            report_name = parts[1]

    tables = soup.find_all("table")
    sections: dict[str, list[str]] = {}
    current_sec = "기본사항"
    sections[current_sec] = []

    for table in tables:
        rows = table.find_all("tr")
        grid: list[list[str | None]] = []

        # Rowspan/Colspan 2D 그리드 복원
        for r_idx, tr in enumerate(rows):
            while len(grid) <= r_idx:
                grid.append([])
            c_idx = 0
            for cell in tr.find_all(["td", "th"]):
                while c_idx < len(grid[r_idx]) and grid[r_idx][c_idx] is not None:
                    c_idx += 1
                rowspan = int(cell.get("rowspan", 1))
                colspan = int(cell.get("colspan", 1))
                txt = clean_txt(cell.get_text(separator=" ", strip=True))

                for r in range(rowspan):
                    while len(grid) <= r_idx + r:
                        grid.append([])
                    for c in range(colspan):
                        while len(grid[r_idx + r]) <= c_idx + c:
                            grid[r_idx + r].append(None)
                        grid[r_idx + r][c_idx + c] = txt if (r == 0 and c == 0) else ""
                c_idx += colspan

        # 그리드 순회 및 섹션 분할
        for row in grid:
            non_empty = [c for c in row if c and c.strip()]
            if not non_empty:
                continue

            first_cell = non_empty[0]
            sec_match = re.match(r"^(\d+[\.\)]\s*[^:]+)$", first_cell)
            if sec_match and len(first_cell) < 80:
                current_sec = sec_match.group(1).strip()
                if current_sec not in sections:
                    sections[current_sec] = []
                rest = non_empty[1:]
            else:
                rest = non_empty

            if not rest:
                continue

            if len(rest) == 1:
                val = re.sub(r"^-\s*", "", rest[0])
                sections[current_sec].append(f"- {val}")
            elif len(rest) == 2:
                k = re.sub(r"^-\s*", "", rest[0])
                v = rest[1]
                sections[current_sec].append(f"- {k}: {v}")
            else:
                sections[current_sec].append(" | ".join(rest))

    valid_sections = {k: v for k, v in sections.items() if v}
    return report_name, company_name, valid_sections


def render_markdown(doc_name: str, company_name: str, sections: dict[str, list[str]]) -> str:
    md_lines = []
    if company_name or doc_name:
        md_lines.append(f"# [{company_name}] {doc_name}".strip())
        md_lines.append("")

    for sec_name, lines in sections.items():
        md_lines.append(f"## {sec_name}")
        md_lines.extend(lines)
        md_lines.append("")

    return "\n".join(md_lines).strip()


def build_chunks(sections: dict[str, list[str]], base_meta: dict) -> list[dict]:
    chunks = []
    idx = 0
    corp_name = base_meta.get("corp_name", "")
    report_nm = base_meta.get("report_nm", "")
    rcept_dt = base_meta.get("rcept_dt", "")
    rcept_no = base_meta.get("rcept_no", "")

    for sec_name, lines in sections.items():
        if not lines:
            continue
        idx += 1
        body = "\n".join(lines).strip()
        header = f"[{corp_name}] {report_nm} ({rcept_dt})".strip()
        text_with_context = f"{header}\n섹션: {sec_name}\n{body}".strip()

        match = re.match(r"^(\d+)", sec_name)
        sec_code = match.group(1) if match else str(idx)

        chunk = {
            **base_meta,
            "chunk_id": f"{rcept_no}_{idx:04d}",
            "chunk_index": idx,
            "chunk_type": "text",
            "section": sec_name,
            "text": text_with_context,
            "n_chars": len(text_with_context),
            # 호환용 추가 필드
            "section_path": sec_name,
            "section_code": sec_code,
            "page": None,
        }
        chunks.append(chunk)

    return chunks


def load_manifest_meta(manifest_path, rcept_no):
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if str(d.get("rcept_no")) == str(rcept_no):
                return d
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xml_path")
    ap.add_argument("manifest_path")
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    xml_path = Path(args.xml_path)
    rcept_no = xml_path.stem

    meta = load_manifest_meta(args.manifest_path, rcept_no)
    if meta is None:
        print(f"[경고] manifest.jsonl 에서 rcept_no={rcept_no} 를 찾지 못했습니다.", file=sys.stderr)
        meta = {}

    doc_name, company_name, sections = extract_exchange_blocks(xml_path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 마크다운 저장
    md = render_markdown(doc_name or meta.get("report_nm", ""), company_name or meta.get("corp_name", ""), sections)
    (out_dir / f"{rcept_no}.md").write_text(md, encoding="utf-8")

    # 공통 스키마 청킹
    doc_group_map = {
        "periodic": "정기공시",
        "major": "주요사항보고서",
        "exchange": "거래소공시",
        "holding": "지분공시",
    }
    base_meta = {
        "source_doc_type": doc_group_map.get(meta.get("doc_group", "exchange"), "거래소공시"),
        "doc_subtype": meta.get("doc_subtype", ""),
        "corp_name": meta.get("corp_name", company_name),
        "corp_code": meta.get("corp_code", ""),
        "stock_code": meta.get("stock_code", ""),
        "industry": meta.get("industry", ""),
        "sector": meta.get("sector", ""),
        "rcept_no": rcept_no,
        "rcept_dt": meta.get("rcept_dt", ""),
        "report_nm": meta.get("report_nm", doc_name),
        "is_correction": meta.get("is_correction", False),
        "source_file": "corpus/" + str(meta.get("file_path", "")) + "/" + xml_path.name,
    }

    chunks = build_chunks(sections, base_meta)
    chunks_path = out_dir / f"{rcept_no}.chunks.jsonl"
    with open(chunks_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"[OK] {rcept_no}: {len(chunks)} chunks -> {out_dir}")


if __name__ == "__main__":
    main()
