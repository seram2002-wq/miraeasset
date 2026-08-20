# -*- coding: utf-8 -*-
"""
DART 정기공시 XML -> Markdown -> Chunk (공통 스키마) 파이프라인
================================================================
담당 서식: periodic (정기공시 - 사업/반기/분기보고서)

사용:
    python3 dart_periodic_chunker.py <xml_path> <manifest.jsonl> [--out-dir DIR]

산출물 (out-dir 기준):
    {rcept_no}.md          - 문서 전체를 재구성한 마크다운 (검수/디버깅용)
    {rcept_no}.chunks.jsonl - 공통 스키마로 청킹된 결과 (임베딩 입력용)
"""

import json
import re
import sys
import argparse
from pathlib import Path
from lxml import etree

# ----------------------------------------------------------------------
# 설정
# ----------------------------------------------------------------------
MAX_CHUNK_CHARS = 1400      # 텍스트 청크 최대 길이 (한국어 기준, 대략 700~900 토큰)
MIN_CHUNK_CHARS = 200        # 이보다 작은 조각은 앞/뒤와 병합 시도
OVERLAP_CHARS = 150          # 텍스트 청크 간 오버랩
TABLE_AS_SINGLE_CHUNK_LIMIT = 3000  # 이 길이 이하 표는 통째로 1개 청크 유지

SECTION_TAGS = {"SECTION-1", "SECTION-2", "SECTION-3", "SECTION-4"}


# ----------------------------------------------------------------------
# 1. 표(TABLE) -> 마크다운 변환 (colspan/rowspan 그리드 전개)
# ----------------------------------------------------------------------
def cell_text(el):
    """TD/TH/TU/TE 셀 내부 텍스트 추출 (하위 P/SPAN 포함, 공백 정규화)."""
    text = "".join(el.itertext())
    text = re.sub(r"\s+", " ", text).strip()
    return text


def table_to_markdown(table_el):
    """TABLE 엘리먼트 하나를 마크다운 표로 변환. rowspan/colspan은 그리드에
    값을 반복 배치하여 표현한다 (병합 셀의 값이 여러 칸에 나타남 = 의미 보존 우선).

    핵심: 각 행을 처리할 때 "이 행의 전체 컬럼 수(n_cols)"를 기준으로 왼쪽부터
    끝까지 훑으면서, 그 위치가 이전 행의 rowspan이 아직 남아있는 칸이면 그 값을
    채우고, 아니면 다음 실제 셀을 꺼내 배치한다. (기존 버전은 "마지막 실제 셀
    뒤에 남은 rowspan 칸"을 채우지 못하는 버그가 있었음 — 헤더 rowspan이 데이터
    행으로 밀려 들어가는 문제가 있어 이번에 수정.)
    """
    rows_src = table_el.findall(".//TR")
    if not rows_src:
        return ""

    # 1) 전체 컬럼 수 결정: COLGROUP 우선, 없으면 각 행의 colspan 합 중 최댓값
    n_cols = 0
    colgroup = table_el.find(".//COLGROUP")
    if colgroup is not None:
        n_cols = len(colgroup.findall("COL"))

    def row_cells(tr):
        out = []
        for cell in tr:
            tag = etree.QName(cell).localname if isinstance(cell.tag, str) else None
            if tag in ("TD", "TH", "TU", "TE"):
                out.append(cell)
        return out

    fallback_max = 0
    for tr in rows_src:
        s = sum(int(c.get("COLSPAN", 1) or 1) for c in row_cells(tr))
        fallback_max = max(fallback_max, s)
    n_cols = max(n_cols, fallback_max)
    if n_cols == 0:
        return ""

    grid = {}          # (row_idx, col_idx) -> text
    pending = {}        # col_idx -> [remaining_rows, value]

    for row_idx, tr in enumerate(rows_src):
        cells = iter(row_cells(tr))
        col_idx = 0
        while col_idx < n_cols:
            p = pending.get(col_idx)
            if p and p[0] > 0:
                grid[(row_idx, col_idx)] = p[1]
                p[0] -= 1
                col_idx += 1
                continue
            try:
                cell = next(cells)
            except StopIteration:
                break  # 행이 n_cols보다 짧게 끝남 (병합 없는 빈 트레일링) -> 남은 칸은 공백
            val = cell_text(cell)
            colspan = int(cell.get("COLSPAN", 1) or 1)
            rowspan = int(cell.get("ROWSPAN", 1) or 1)
            for c in range(colspan):
                if col_idx + c >= n_cols:
                    break
                grid[(row_idx, col_idx + c)] = val
                if rowspan > 1:
                    pending[col_idx + c] = [rowspan - 1, val]
            col_idx += colspan

    n_rows = len(rows_src)
    if n_rows == 0 or n_cols == 0:
        return ""

    max_col = n_cols
    matrix = [[grid.get((r, c), "") for c in range(max_col)] for r in range(n_rows)]

    # 완전히 빈 행 제거
    matrix = [r for r in matrix if any(v.strip() for v in r)]
    if not matrix:
        return ""

    def esc(v):
        return v.replace("|", "\\|").replace("\n", " ").strip() or " "

    lines = []
    header = matrix[0]
    lines.append("| " + " | ".join(esc(v) for v in header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for r in matrix[1:]:
        lines.append("| " + " | ".join(esc(v) for v in r) + " |")

    return "\n".join(lines)


# ----------------------------------------------------------------------
# 2. XML 트리 순회 -> (섹션 경로, 블록) 리스트
# ----------------------------------------------------------------------
def local(tag):
    if not isinstance(tag, str):
        return ""
    return etree.QName(tag).localname


def walk(el, section_stack, blocks, in_table=False):
    """el 하위를 문서 순서대로 순회하며 blocks 리스트에
    {'type': 'text'|'table', 'text': str, 'section_path': [...]} 를 채운다."""
    tag = local(el.tag)

    if tag in SECTION_TAGS:
        level = int(tag.split("-")[1])
        # 해당 레벨 이상 스택 정리
        while section_stack and section_stack[-1][0] >= level:
            section_stack.pop()
        title_el = el.find("TITLE")
        title = cell_text(title_el) if title_el is not None else ""
        section_stack.append([level, title])
        for child in el:
            walk(child, section_stack, blocks, in_table)
        return

    if tag == "TABLE":
        md = table_to_markdown(el)
        if md:
            blocks.append({
                "type": "table",
                "text": md,
                "section_path": [t for _, t in section_stack if t],
            })
        return  # 표 내부는 더 이상 순회하지 않음 (중복 방지)

    if tag == "P" and not in_table:
        txt = cell_text(el)
        if txt:
            blocks.append({
                "type": "text",
                "text": txt,
                "section_path": [t for _, t in section_stack if t],
            })
        return

    if tag == "TITLE":
        return  # 이미 section_stack 처리 시 사용함

    # 그 외 컨테이너 태그는 계속 하위 순회
    for child in el:
        walk(child, section_stack, blocks, in_table)


def extract_blocks(xml_path):
    parser = etree.XMLParser(recover=True, huge_tree=True)
    tree = etree.parse(str(xml_path), parser)
    root = tree.getroot()

    doc_name_el = root.find(".//DOCUMENT-NAME")
    company_el = root.find(".//COMPANY-NAME")
    doc_name = cell_text(doc_name_el) if doc_name_el is not None else ""
    company_name = cell_text(company_el) if company_el is not None else ""

    body = root.find(".//BODY")
    blocks = []
    if body is not None:
        section_stack = []
        for child in body:
            walk(child, section_stack, blocks)

    return doc_name, company_name, blocks


# ----------------------------------------------------------------------
# 3. blocks -> 마크다운 전체 문서 렌더링 (검수용)
# ----------------------------------------------------------------------
def render_markdown(doc_name, company_name, blocks):
    lines = [f"# {company_name} - {doc_name}", ""]
    last_path = []
    for b in blocks:
        path = b["section_path"]
        if path != last_path:
            # 새 섹션이면 헤더 출력 (path 길이에 따라 #, ##, ### ...)
            common = 0
            while (common < len(path) and common < len(last_path)
                   and path[common] == last_path[common]):
                common += 1
            for i in range(common, len(path)):
                level = min(i + 2, 6)
                lines.append("")
                lines.append("#" * level + " " + path[i])
            last_path = path
        if b["type"] == "table":
            lines.append("")
            lines.append(b["text"])
            lines.append("")
        else:
            lines.append(b["text"])
    return "\n".join(lines)


# ----------------------------------------------------------------------
# 4. blocks -> 공통 스키마 청크
# ----------------------------------------------------------------------
def split_long_text(text, max_chars=MAX_CHUNK_CHARS, overlap=OVERLAP_CHARS):
    """긴 텍스트를 문장 경계 위주로 max_chars 이하 조각으로 분할 (오버랩 포함)."""
    if len(text) <= max_chars:
        return [text]
    # 문장 경계(마침표, 개행) 기준 우선 분할
    sentences = re.split(r"(?<=[.?!다음:])\s+", text)
    chunks = []
    cur = ""
    for s in sentences:
        if len(cur) + len(s) + 1 <= max_chars:
            cur = (cur + " " + s).strip()
        else:
            if cur:
                chunks.append(cur)
            # 오버랩: 이전 청크 끝부분을 새 청크 시작에 붙임
            tail = cur[-overlap:] if cur else ""
            cur = (tail + " " + s).strip()
            # 문장 자체가 max_chars보다 길면 강제 분할
            while len(cur) > max_chars:
                chunks.append(cur[:max_chars])
                cur = cur[max_chars - overlap:]
    if cur:
        chunks.append(cur)
    return chunks


def build_chunks(blocks, base_meta):
    """섹션 단위로 텍스트를 묶고, 표는 원자적으로 유지하며 청킹."""
    chunks = []
    idx = 0

    def flush(buf_text, section_path):
        nonlocal idx
        if not buf_text.strip():
            return
        for piece in split_long_text(buf_text.strip()):
            idx += 1
            chunks.append({
                **base_meta,
                "chunk_id": f"{base_meta['rcept_no']}_{idx:04d}",
                "chunk_index": idx,
                "chunk_type": "text",
                "section": " > ".join(section_path) if section_path else "",
                "text": piece,
                "n_chars": len(piece),
            })

    buf = ""
    buf_path = None

    for b in blocks:
        path = b["section_path"]
        if b["type"] == "table":
            # 표를 만나면 지금까지 쌓인 텍스트 버퍼를 먼저 flush
            flush(buf, buf_path or path)
            buf, buf_path = "", None

            table_text = b["text"]
            idx += 1
            if len(table_text) <= TABLE_AS_SINGLE_CHUNK_LIMIT:
                chunks.append({
                    **base_meta,
                    "chunk_id": f"{base_meta['rcept_no']}_{idx:04d}",
                    "chunk_index": idx,
                    "chunk_type": "table",
                    "section": " > ".join(path) if path else "",
                    "text": table_text,
                    "n_chars": len(table_text),
                })
            else:
                # 큰 표는 헤더(첫 2줄: 헤더+구분선)를 유지한 채 행 단위로 분할
                lines = table_text.split("\n")
                header = lines[:2]
                body_lines = lines[2:]
                cur_lines = list(header)
                cur_len = sum(len(l) for l in cur_lines)
                part = 1
                for line in body_lines:
                    if cur_len + len(line) > TABLE_AS_SINGLE_CHUNK_LIMIT and len(cur_lines) > 2:
                        chunks.append({
                            **base_meta,
                            "chunk_id": f"{base_meta['rcept_no']}_{idx:04d}_p{part}",
                            "chunk_index": idx,
                            "chunk_type": "table",
                            "section": " > ".join(path) if path else "",
                            "text": "\n".join(cur_lines),
                            "n_chars": sum(len(l) for l in cur_lines),
                        })
                        part += 1
                        cur_lines = list(header)
                        cur_len = sum(len(l) for l in cur_lines)
                    cur_lines.append(line)
                    cur_len += len(line)
                if len(cur_lines) > 2:
                    chunks.append({
                        **base_meta,
                        "chunk_id": f"{base_meta['rcept_no']}_{idx:04d}_p{part}",
                        "chunk_index": idx,
                        "chunk_type": "table",
                        "section": " > ".join(path) if path else "",
                        "text": "\n".join(cur_lines),
                        "n_chars": sum(len(l) for l in cur_lines),
                    })
            continue

        # type == text
        if buf_path is not None and path != buf_path:
            flush(buf, buf_path)
            buf = ""
        buf_path = path
        buf = (buf + "\n" + b["text"]).strip()
        if len(buf) > MAX_CHUNK_CHARS:
            flush(buf, buf_path)
            buf = ""

    flush(buf, buf_path)
    return chunks


# ----------------------------------------------------------------------
# 5. manifest.jsonl 에서 메타데이터 로드
# ----------------------------------------------------------------------
def load_manifest_meta(manifest_path, rcept_no):
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("rcept_no") == rcept_no:
                return d
    return None


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xml_path")
    ap.add_argument("manifest_path")
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    xml_path = Path(args.xml_path)
    rcept_no = xml_path.stem  # 파일명 = 접수번호

    meta = load_manifest_meta(args.manifest_path, rcept_no)
    if meta is None:
        print(f"[경고] manifest.jsonl 에서 rcept_no={rcept_no} 를 찾지 못했습니다. "
              f"파일명/도메인 정보만으로 진행합니다.", file=sys.stderr)
        meta = {}

    doc_name, company_name, blocks = extract_blocks(xml_path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 마크다운 저장 ---
    md = render_markdown(doc_name, company_name or meta.get("corp_name", ""), blocks)
    md_path = out_dir / f"{rcept_no}.md"
    md_path.write_text(md, encoding="utf-8")

    # --- 공통 스키마 청킹 ---
    doc_group_map = {
        "periodic": "정기공시",
        "major": "주요사항보고서",
        "exchange": "거래소공시",
        "holding": "지분공시",
    }
    base_meta = {
        "source_doc_type": doc_group_map.get(meta.get("doc_group", "periodic"), "정기공시"),
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
        "base_year": meta.get("base_year"),
        "base_month": meta.get("base_month"),
        "source_file": meta.get("file_path", str(xml_path)) + f"/{xml_path.name}",
    }

    chunks = build_chunks(blocks, base_meta)

    chunks_path = out_dir / f"{rcept_no}.chunks.jsonl"
    with open(chunks_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"완료: {len(blocks)} blocks -> {len(chunks)} chunks")
    print(f"  마크다운: {md_path}")
    print(f"  청크(JSONL): {chunks_path}")


if __name__ == "__main__":
    main()
