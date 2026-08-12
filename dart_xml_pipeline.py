# -*- coding: utf-8 -*-
"""
dart_xml_pipeline.py
=====================================================================
DART(전자공시시스템) 원문 XML(dart3.xsd 스키마) 일괄 처리 파이프라인
  - 대상: 지분공시(주식등의대량보유상황보고서), 주요사항보고서 등
          <DOCUMENT>...<BODY><SECTION-n>...<TABLE>...</BODY></DOCUMENT>
          구조를 갖는 모든 DART 서식(정기공시 포함 범용). 거래소 공시는 안됨.

  - 실행방법: 주의: run 버튼 누르면 안되고 터미널에 parameter 입력해야함. main 함수 참고.

          python3 (파이썬 파일있는 경로)/dart_xml_pipeline.py batch(만약 한개만 실행하고 싶다면 one 입력) \
  "(해당 jsonl 파일 위치)/manifest.jsonl" \
  "(corpus 파일 위치)/corpus" \
  "(저장하고 싶은 위치)/output" \
  --doc_group (major, periodic 등으로 변경 가능)holding
  
    
      예시)  python3 /Users/chanuyoung/dart_xml_pipeline.py batch \
  "/Users/chanuyoung/Documents/2026/summer_intership/contest/mirae/gongsi/corpus/manifest.jsonl"\
        "/Users/chanuyoung/Documents/2026/summer_intership/contest/mirae/gongsi/corpus" \
  "/Users/chanuyoung/Documents/2026/summer_intership/contest/mirae/gongsi/corpus/output" \
  --doc_group exchange

3단계 처리
---------------------------------------------------------------------
[1단계] extract_raw(path)
    원문 XML을 그대로 순회 → 문서 메타정보 + 섹션 트리(SECTION-1~4) +
    표(TABLE, ROWSPAN/COLSPAN 격자 복원) + 문단(P)을 하나도 버리지 않고
    구조 그대로 dict 로 추출한다. (DART 서식필드 코드 ACODE/AUNIT 보존)

[2단계] normalize(raw)
    1단계 dict를 입력받아
      - 표 안의 라벨-값 쌍을 (section_path, label, acode, value, value_type)
        형태의 flat 필드 리스트로 변환
      - 숫자("1,234,567" → 1234567), 날짜("2023년 08월 30일"/AUNITVALUE
        "20230830" → "2023-08-30"), 결측("-","해당없음" 등) → None 처리
      - 문서 레벨 메타(정정공시 여부, 법인명 클린업 등) 정리
    를 수행한다.

[3단계] to_markdown(normalized)
    정규화 결과의 섹션 트리를 사람이 읽기 좋은 마크다운 문서
    (헤딩 + GFM 표 + 문단)로 직렬화한다. LLM 입력/RAG 용도로 바로 사용 가능.

일괄 처리
---------------------------------------------------------------------
batch_process(manifest_path, corpus_root, out_dir, ...)
    manifest.jsonl 의 메타데이터를 기준으로 raw/ 하위 XML을 모두 찾아
    위 3단계를 적용하고, 문서별로
        {out_dir}/raw/{doc_id}.json
        {out_dir}/normalized/{doc_id}.json
        {out_dir}/markdown/{doc_id}.md
    을 생성한다. doc_group/doc_subtype/기간 등으로 필터링 가능.

의존성: lxml (pip install lxml)
---------------------------------------------------------------------
"""
from __future__ import annotations

import json
import re
import sys
import traceback
from pathlib import Path
from typing import Iterable, Optional

from lxml import etree

# =====================================================================
# 공통 상수
# =====================================================================

# DART 원문 XML은 종종 "&"가 이스케이프되지 않은 채로 들어있다 (예: "Ernst&Young").
# 표준 XML 파서는 이를 만나면 실패하므로, 파싱 전에 안전하게 escape 처리한다.
_AMP_RE = re.compile(rb'&(?!amp;|lt;|gt;|quot;|apos;|#\d+;|#x[0-9a-fA-F]+;)')
_CTRL_RE = re.compile(rb'[\x00-\x08\x0b\x0c\x0e-\x1f]')  # XML 1.0에서 허용되지 않는 제어문자

_CONTAINER_TAGS = {'BODY', 'LIBRARY', 'COVER', 'SECTION-1', 'SECTION-2', 'SECTION-3', 'SECTION-4'}
_VALUE_TAGS = {'TE', 'TU'}   # TE: 추출필드(ACODE) / TU: 코드화된 선택값(AUNIT+AUNITVALUE)
_LABEL_TAGS = {'TD', 'TH'}   # 서식상의 라벨(항목명) 셀

_SECTION_LEVEL = {
    'BODY': 0, 'LIBRARY': 1, 'COVER': 1,
    'SECTION-1': 1, 'SECTION-2': 2, 'SECTION-3': 3, 'SECTION-4': 4,
}

_NUMERIC_RE = re.compile(r'^-?\d{1,3}(,\d{3})*(\.\d+)?$')
_DATE8_RE = re.compile(r'^\d{8}$')
_EMPTY_VALUES = {'', '-', '–', 'N/A', '해당없음', '해당사항없음', '해당 없음'}


# =====================================================================
# 0. XML 로딩 (오염된 원문 대응)
# =====================================================================

def sanitize_xml_bytes(raw: bytes) -> bytes:
    """DART 원문 XML에서 흔히 발견되는 미이스케이프 '&', 제어문자를 정리한다."""
    raw = _CTRL_RE.sub(b'', raw)
    raw = _AMP_RE.sub(b'&amp;', raw)
    return raw


def parse_dart_xml(path) -> etree._Element:
    """DART document.xml 원문을 lxml Element 트리로 로드한다.
    recover=True 로 일부 태그 불일치 등 경미한 오류도 최대한 복구한다."""
    raw = Path(path).read_bytes()
    raw = sanitize_xml_bytes(raw)
    parser = etree.XMLParser(recover=True, huge_tree=True)
    root = etree.fromstring(raw, parser=parser)
    if root is None:
        raise ValueError(f'XML 파싱 실패 (복구 불가): {path}')
    return root


# =====================================================================
# 1단계: 원문 필드 추출 (extract_raw)
# =====================================================================

def _cell_text(el) -> str:
    text = ''.join(el.itertext())
    text = text.replace('\xa0', ' ').replace('\u3000', ' ')
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n+', '\n', text)
    return text.strip()


def table_to_grid(table_el) -> list[list[dict]]:
    """<TABLE> 하위 <TR>/<TD|TE|TU|TH>를 ROWSPAN/COLSPAN까지 반영한
    2차원 격자(list[row][col] = cell dict)로 복원한다.
    cell dict: {tag, text, acode, aunit, aunitvalue}
    ※ ROWSPAN 으로 확장된 칸들은 동일 cell(dict) 객체를 참조 → id() 로 원본 식별 가능."""
    rows_el = table_el.xpath('.//TR')
    grid = []
    active: dict[int, list] = {}  # col -> [남은 확장행수, cell]
    for tr in rows_el:
        row_map = {c: cell for c, (_, cell) in active.items()}
        children = [ch for ch in tr if ch.tag in ('TD', 'TE', 'TU', 'TH')]
        cur_col = 0
        new_spans = {}
        for ch in children:
            while cur_col in row_map and cur_col not in new_spans:
                cur_col += 1
            try:
                colspan = int(ch.get('COLSPAN') or 1)
            except ValueError:
                colspan = 1
            try:
                rowspan = int(ch.get('ROWSPAN') or 1)
            except ValueError:
                rowspan = 1
            cell = {
                'tag': ch.tag,
                'text': _cell_text(ch),
                'acode': ch.get('ACODE'),
                'aunit': ch.get('AUNIT'),
                'aunitvalue': ch.get('AUNITVALUE'),
            }
            for cc in range(cur_col, cur_col + colspan):
                row_map[cc] = cell
                if rowspan > 1:
                    new_spans[cc] = rowspan - 1
            cur_col += colspan
        maxc = (max(row_map.keys()) + 1) if row_map else 0
        row = [row_map.get(i, {'tag': None, 'text': '', 'acode': None, 'aunit': None, 'aunitvalue': None})
               for i in range(maxc)]
        grid.append(row)
        next_active = {}
        for c, (rem, cell) in active.items():
            if rem - 1 > 0:
                next_active[c] = [rem - 1, cell]
        for c, rem in new_spans.items():
            next_active[c] = [rem, row_map[c]]
        active = next_active
    return grid


def _fields_from_grid(grid: list[list[dict]]) -> list[dict]:
    """격자에서 (라벨, 값) 필드 레코드를 추출한다.
    한 행 안에서 라벨(TD/TH)이 연속으로 나오면 하나의 복합 라벨로 합치고,
    그 다음에 나오는 값 셀(TE/TU)에 매칭한다 (DART 서식의 전형적 패턴).
    ROWSPAN 확장으로 같은 셀이 여러 행에 중복 등장하는 것은 id() 로 1회만 채택."""
    fields, seen = [], set()
    for row in grid:
        label_parts = []
        for cell in row:
            if cell['tag'] in _LABEL_TAGS and cell['text']:
                label_parts.append(cell['text'])
            elif cell['tag'] in _VALUE_TAGS:
                key = id(cell)
                if key in seen:
                    continue
                seen.add(key)
                fields.append({
                    'label': ' / '.join(dict.fromkeys(label_parts)) if label_parts else None,
                    'tag': cell['tag'],
                    'acode': cell['acode'],
                    'aunit': cell['aunit'],
                    'aunitvalue': cell['aunitvalue'],
                    'value': cell['text'],
                })
                label_parts = []
    return fields


def _walk(el) -> list[dict]:
    """컨테이너(BODY/LIBRARY/COVER/SECTION-n) 하위를 문서 순서대로 재귀 순회하며
    section/paragraph/table 블록 리스트를 만든다."""
    blocks = []
    for ch in el:
        tag = ch.tag
        if tag in _CONTAINER_TAGS:
            title_el = ch.find('TITLE')
            blocks.append({
                'type': 'section',
                'tag': tag,
                'level': _SECTION_LEVEL.get(tag, 9),
                'title': _cell_text(title_el) if title_el is not None else None,
                'assocnote': title_el.get('AASSOCNOTE') if title_el is not None else None,
                'children': _walk(ch),
            })
        elif tag == 'TABLE-GROUP':
            for tbl in ch.findall('TABLE'):
                grid = table_to_grid(tbl)
                blocks.append({'type': 'table', 'aclass': ch.get('ACLASS'),
                                'grid': grid, 'fields': _fields_from_grid(grid)})
        elif tag == 'TABLE':
            grid = table_to_grid(ch)
            blocks.append({'type': 'table', 'aclass': None,
                            'grid': grid, 'fields': _fields_from_grid(grid)})
        elif tag == 'P':
            text = _cell_text(ch)
            if text:
                blocks.append({'type': 'paragraph', 'text': text})
        elif tag in ('PGBRK', 'TITLE'):
            continue
        else:
            # SPAN 등 알려지지 않은 태그는 텍스트만 문단으로 흡수 (내용 유실 방지)
            text = _cell_text(ch)
            if text:
                blocks.append({'type': 'paragraph', 'text': text})
    return blocks


def extract_raw(path) -> dict:
    """[1단계] 원문 필드 추출.
    반환값: {source_file, document_name, document_acode, formula_version,
             company_name, company_regcik, summary, sections}"""
    root = parse_dart_xml(path)
    doc_name_el = root.find('DOCUMENT-NAME')
    fv_el = root.find('FORMULA-VERSION')
    comp_el = root.find('COMPANY-NAME')
    summary_el = root.find('SUMMARY')
    summary = []
    if summary_el is not None:
        for ex in summary_el.findall('EXTRACTION'):
            summary.append({'acode': ex.get('ACODE'), 'afeature': ex.get('AFEATURE'), 'value': _cell_text(ex)})
    body_el = root.find('BODY')
    return {
        'source_file': str(path),
        'document_name': _cell_text(doc_name_el) if doc_name_el is not None else None,
        'document_acode': doc_name_el.get('ACODE') if doc_name_el is not None else None,
        'formula_version': _cell_text(fv_el) if fv_el is not None else None,
        'company_name': _cell_text(comp_el) if comp_el is not None else None,
        'company_regcik': comp_el.get('AREGCIK') if comp_el is not None else None,
        'summary': summary,
        'sections': _walk(body_el) if body_el is not None else [],
    }


# =====================================================================
# 2단계: 정규화 (normalize)
# =====================================================================

def normalize_scalar(text: Optional[str], aunitvalue: Optional[str] = None):
    """값 하나를 (정규화된 값, 타입) 으로 변환한다.
    타입: 'null' | 'empty' | 'date' | 'number' | 'text'"""
    if text is None:
        return None, 'null'
    t = text.strip()
    if t in _EMPTY_VALUES:
        return None, 'empty'
    # 날짜: TU 요소는 AUNITVALUE 에 YYYYMMDD 원시코드가 들어있는 경우가 많음(가장 신뢰도 높음)
    if aunitvalue and _DATE8_RE.match(aunitvalue):
        return f'{aunitvalue[:4]}-{aunitvalue[4:6]}-{aunitvalue[6:8]}', 'date'
    if _DATE8_RE.match(t) and t[:2] in ('19', '20'):
        return f'{t[:4]}-{t[4:6]}-{t[6:8]}', 'date'
    # 숫자: "1,234,567" / "-63,993" / "46.85"
    if _NUMERIC_RE.match(t):
        cleaned = t.replace(',', '')
        try:
            return (float(cleaned) if '.' in cleaned else int(cleaned)), 'number'
        except ValueError:
            pass
    return t, 'text'


def _normalize_walk(blocks, breadcrumb, fields_out, paragraphs_out):
    for b in blocks:
        if b['type'] == 'section':
            crumb = breadcrumb + ([b['title']] if b['title'] else [])
            _normalize_walk(b['children'], crumb, fields_out, paragraphs_out)
        elif b['type'] == 'paragraph':
            paragraphs_out.append({'section_path': breadcrumb, 'text': b['text']})
        elif b['type'] == 'table':
            for f in b['fields']:
                value, vtype = normalize_scalar(f['value'], f.get('aunitvalue'))
                fields_out.append({
                    'section_path': breadcrumb,
                    'table_aclass': b['aclass'],
                    'label': f['label'],
                    'tag': f['tag'],
                    'acode': f['acode'],
                    'aunit': f['aunit'],
                    'aunitvalue': f['aunitvalue'],
                    'raw_value': f['value'],
                    'value': value,
                    'value_type': vtype,
                })


def normalize(raw: dict) -> dict:
    """[2단계] 정규화.
    반환값: {meta, summary, fields(flat), paragraphs(flat), sections(마크다운 변환용 원본 트리)}"""
    doc_name = raw.get('document_name') or ''
    is_correction = doc_name.strip().startswith('[기재정정]')
    company_name = (raw.get('company_name') or '').strip()
    company_name_clean = re.sub(r'\s*\(주\)\s*$|^\s*\(주\)\s*', '', company_name).strip()

    fields, paragraphs = [], []
    _normalize_walk(raw.get('sections', []), [], fields, paragraphs)

    return {
        'meta': {
            'document_name': doc_name,
            'document_name_clean': doc_name.replace('[기재정정]', '').strip(),
            'is_correction': is_correction,
            'document_acode': raw.get('document_acode'),
            'company_name': company_name,
            'company_name_clean': company_name_clean,
            'company_regcik': raw.get('company_regcik'),
            'source_file': raw.get('source_file'),
        },
        'summary': raw.get('summary', []),
        'fields': fields,
        'paragraphs': paragraphs,
        'sections': raw.get('sections', []),
    }


# =====================================================================
# 3단계: 텍스트화 / 마크다운화 (to_markdown)
# =====================================================================

def _grid_to_md_table(grid: list[list[dict]]) -> Optional[str]:
    """격자를 GFM 마크다운 표로 직렬화한다.
    ROWSPAN/COLSPAN 으로 여러 칸에 걸친 값은 table_to_grid 단계에서 동일 셀(dict) 객체가
    여러 grid 위치에 중복 참조되는데, 그대로 텍스트화하면 "같은 값이 여러 칸에 반복"되어
    보여서 혼란스럽다. 따라서 각 셀은 grid 상에서 처음 등장하는 위치(좌상단)에서만 값을 쓰고
    나머지 병합 위치는 빈 칸으로 둔다(스프레드시트의 '병합된 셀' 표시 방식과 동일).
    단, DART 서식이 정말로 폭 맞춤을 위해 동일 텍스트의 별개 라벨 셀을 나란히 배치한 경우
    (동일 문자열이지만 서로 다른 XML 요소)는 이 로직으로 합쳐지지 않는다 — 원문 그대로다."""
    def clean(t: str) -> str:
        return t.replace('\n', '<br>').replace('|', '\\|')

    printed_ids: set[int] = set()

    def cell_text(cell: dict) -> str:
        key = id(cell)
        if key in printed_ids:
            return ''
        printed_ids.add(key)
        return clean(cell['text'])

    header_rows, body_start = [], 0
    for row in grid:
        if any(c['tag'] in _VALUE_TAGS for c in row):
            break
        header_rows.append(row)
        body_start += 1
    body_rows = grid[body_start:]

    if header_rows and body_rows:
        width = max(len(r) for r in header_rows + body_rows)
        merged_header = []
        for col in range(width):
            parts = []
            for r in header_rows:
                if col < len(r):
                    txt = r[col]['text'].strip()
                    if txt and (not parts or parts[-1] != txt):  # 동일 텍스트 반복(rowspan) 압축
                        parts.append(txt)
            merged_header.append(clean(' '.join(parts)))
            for r in header_rows:  # 헤더 셀도 병합-중복 표시 방지 대상으로 등록
                if col < len(r):
                    printed_ids.add(id(r[col]))
        body_text_rows = [[cell_text(c) for c in r] for r in body_rows]
        body_text_rows = [r for r in body_text_rows if any(c.strip() for c in r)]
        if not body_text_rows:
            return None
        width = max(width, max(len(r) for r in body_text_rows))
        merged_header += [''] * (width - len(merged_header))
        body_text_rows = [r + [''] * (width - len(r)) for r in body_text_rows]
        lines = ['| ' + ' | '.join(merged_header) + ' |',
                 '| ' + ' | '.join(['---'] * width) + ' |']
        lines += ['| ' + ' | '.join(r) + ' |' for r in body_text_rows]
        return '\n'.join(lines)

    # 값 셀이 전혀 없는 표(순수 안내문 등) 또는 헤더 행이 없는 표: 첫 행을 헤더로 사용
    rows = [[cell_text(c) for c in row] for row in grid]
    rows = [r for r in rows if any(c.strip() for c in r)]
    if not rows:
        return None
    if len(rows) == 1 and len(rows[0]) == 1:
        return f'> {rows[0][0]}' if rows[0][0] else None
    width = max(len(r) for r in rows)
    rows = [r + [''] * (width - len(r)) for r in rows]
    lines = ['| ' + ' | '.join(rows[0]) + ' |',
             '| ' + ' | '.join(['---'] * width) + ' |']
    lines += ['| ' + ' | '.join(r) + ' |' for r in rows[1:]]
    return '\n'.join(lines)


def _md_walk(blocks, level, lines):
    for b in blocks:
        if b['type'] == 'section':
            if b['title']:
                lines.append('#' * min(level + 1, 6) + ' ' + b['title'])
            _md_walk(b['children'], level + (1 if b['title'] else 0), lines)
        elif b['type'] == 'paragraph':
            lines.append(b['text'])
        elif b['type'] == 'table':
            md = _grid_to_md_table(b['grid'])
            if md:
                lines.append(md)


def to_markdown(normalized: dict) -> str:
    """[3단계] 텍스트화(마크다운화). normalize() 결과를 입력으로 받는다."""
    meta = normalized['meta']
    lines = [f"# {meta['document_name']}", f"**제출인/회사명:** {meta['company_name']}"]
    if meta['is_correction']:
        lines.insert(1, '> ⚠️ 본 문서는 [기재정정] 정정공시입니다.')
    _md_walk(normalized['sections'], level=1, lines=lines)
    text = '\n\n'.join(l for l in lines if l is not None)
    return re.sub(r'\n{3,}', '\n\n', text)


# =====================================================================
# 단일 문서 처리 헬퍼 + 배치 처리
# =====================================================================

def process_document(path) -> dict:
    """XML 한 건에 3단계를 모두 적용해 {raw, normalized, markdown} 을 반환."""
    raw = extract_raw(path)
    normalized = normalize(raw)
    markdown = to_markdown(normalized)
    return {'raw': raw, 'normalized': normalized, 'markdown': markdown}


def iter_manifest(manifest_path, **filters) -> Iterable[dict]:
    """manifest.jsonl 을 순회하며 filters(예: doc_group='holding')에 맞는 레코드만 yield.
    filters 값이 callable 이면 predicate 로, 아니면 등가비교로 취급."""
    with open(manifest_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ok = True
            for k, v in filters.items():
                rv = rec.get(k)
                ok = v(rv) if callable(v) else (rv == v)
                if not ok:
                    break
            if ok:
                yield rec


def batch_process(manifest_path, corpus_root, out_dir, save_raw=True,
                   save_normalized=True, save_markdown=True, on_error='warn', **filters):
    """manifest.jsonl 기준으로 raw/ 하위 XML을 전부 찾아 3단계 파이프라인을 적용하고
    out_dir/{raw,normalized,markdown}/ 아래에 문서별 결과를 저장한다.

    Parameters
    ----------
    manifest_path : manifest.jsonl 경로
    corpus_root   : README.md 기준 raw/ 를 포함하는 코퍼스 루트 (file_path의 기준 디렉터리)
    out_dir       : 결과 저장 디렉터리
    filters       : iter_manifest 와 동일 (예: doc_group='holding', doc_group=lambda g: g in {'holding','major'})
    on_error      : 'warn'(기본, 로그만 남기고 계속) | 'raise'

    Returns
    -------
    (성공건수, 실패건수) 및 실패 로그는 stderr 로 출력.
    """
    corpus_root = Path(corpus_root)
    out_dir = Path(out_dir)
    dirs = {}
    for key, flag in (('raw', save_raw), ('normalized', save_normalized), ('markdown', save_markdown)):
        if flag:
            d = out_dir / key
            d.mkdir(parents=True, exist_ok=True)
            dirs[key] = d

    n_ok, n_fail = 0, 0
    for rec in iter_manifest(manifest_path, **filters):
        doc_id = rec['doc_id']
        doc_dir = corpus_root / rec['file_path']
        xml_files = sorted(doc_dir.glob('*.xml')) if doc_dir.exists() else []
        if not xml_files:
            n_fail += 1
            print(f'[SKIP] xml 없음: {doc_id} ({doc_dir})', file=sys.stderr)
            continue
        for xi, xml_path in enumerate(xml_files):
            suffix = doc_id if len(xml_files) == 1 else f'{doc_id}_{xi}'
            try:
                result = process_document(xml_path)
                if 'raw' in dirs:
                    (dirs['raw'] / f'{suffix}.json').write_text(
                        json.dumps(result['raw'], ensure_ascii=False, indent=2), encoding='utf-8')
                if 'normalized' in dirs:
                    norm = result['normalized']
                    payload = {**norm, 'doc_id': doc_id, **{f'manifest_{k}': v for k, v in rec.items()}}
                    (dirs['normalized'] / f'{suffix}.json').write_text(
                        json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
                if 'markdown' in dirs:
                    (dirs['markdown'] / f'{suffix}.md').write_text(result['markdown'], encoding='utf-8')
                n_ok += 1
            except Exception as e:
                n_fail += 1
                msg = f'[FAIL] {doc_id} ({xml_path}): {e}'
                if on_error == 'raise':
                    raise
                print(msg, file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
    print(f'batch_process 완료: 성공 {n_ok}건 / 실패 {n_fail}건', file=sys.stderr)
    return n_ok, n_fail


# =====================================================================
# CLI 
# =====================================================================
if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser(description='DART 공시 XML 3단계 처리 파이프라인')
    sub = ap.add_subparsers(dest='cmd', required=True)

    p1 = sub.add_parser('one', help='단일 XML 파일 처리 (raw/normalized/markdown 콘솔 출력)')
    p1.add_argument('xml_path')

    p2 = sub.add_parser('batch', help='manifest.jsonl 기준 일괄 처리')
    p2.add_argument('manifest_path')
    p2.add_argument('corpus_root')
    p2.add_argument('out_dir')
    p2.add_argument('--doc_group', default=None, help='예: holding, major, periodic, exchange')

    args = ap.parse_args()

    if args.cmd == 'one':
        result = process_document(args.xml_path)
        print(f"필드 {len(result['normalized']['fields'])}건, "
              f"문단 {len(result['normalized']['paragraphs'])}건 추출")
        print('--- markdown 미리보기 (앞 1000자) ---')
        print(result['markdown'][:1000])
    elif args.cmd == 'batch':
        kwargs = {}
        if args.doc_group:
            kwargs['doc_group'] = args.doc_group
        batch_process(args.manifest_path, args.corpus_root, args.out_dir, **kwargs)
