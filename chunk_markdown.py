# -*- coding: utf-8 -*-
"""
chunk_markdown.py
=====================================================================
DART 공시 마크다운(dart_xml_pipeline.py 의 to_markdown() 출력물 등,
헤딩(#) + GFM 파이프 표(|...|) + 문단으로 구성된 마크다운)을
RAG/LLM 입력용으로 800~1000자 단위 청크로 분할한다.

핵심 원칙
---------------------------------------------------------------------
1. 표(GFM 파이프 테이블)는 "행 단위"가 최소 단위다.
   - 표 한 행이 청크 중간에서 잘리는 일은 절대 없다.
   - 표 전체가 목표 크기(max_chars) 안에 들어가면 표를 통째로 한 청크에 담는다.
   - 표가 너무 커서 혼자서도 max_chars를 초과하면, 헤더(+구분행)를
     각 조각에 반복 삽입하면서 "완전한 행" 단위로만 나눈다.
2. 일반 문단/문장은 min_chars~max_chars 구간을 채우도록 그리디하게 합치고,
   그 구간을 넘기지 않고는 더 채울 수 없을 때 청크를 닫는다.
   문단 하나가 max_chars보다 크면 문장 경계(마침표/개행)로만 추가 분할한다.
3. 헤딩(#, ##, ...)은 가능하면 뒤따르는 내용과 같은 청크에 붙인다
   (청크 맨 끝에 헤딩만 덩그러니 남는 것을 피함).
4. 모든 청크는 자신이 속한 섹션 경로(heading_path)를 메타데이터로 갖는다
   (검색/재순위 시 문맥 파악용).

임베딩 파이프라인용 추가 필드
---------------------------------------------------------------------
- `chunk_id`      : "{doc_id}_{chunk_index}" 형태의 전역 고유 ID.
                    여러 문서의 청크를 하나의 벡터DB 컬렉션에 합쳐 넣어도
                    충돌하지 않는다 (upsert 시 PK로 사용).
- `embedding_text`: 임베딩에 실제로 넣을 텍스트. `heading_path`(섹션 경로)를
                    맨 앞에 붙여서, 표만 뚝 잘린 청크도 "이게 어떤 표인지"
                    맥락을 잃지 않게 한다. 원본 `text`는 그대로 보존
                    (화면 표시/재구성용), `embedding_text`만 임베딩 API에 넣으면 됨.
- manifest 조인   : `batch` 명령에 `--manifest manifest.jsonl` 을 주면
                    doc_id 기준으로 corp_name/sector/doc_group/rcept_dt 등의
                    문서 메타데이터를 각 청크 레코드에 합쳐 넣는다
                    (벡터DB 메타데이터 필터링 검색용, 예: "네이버 + 2023년만").

사용법
---------------------------------------------------------------------
  단일 파일:
    python3 chunk_markdown.py one "major_20230102000286.md" --min 800 --max 1000

  디렉터리 일괄 처리 (예: dart_xml_pipeline.py 의 out_dir/markdown/*.md):
 
 터미널 입력 예시)

  python3 /Users/chanuyoung/chunk_markdown.py batch\
      "/Users/chanuyoung/Documents/2026/summer_intership/contest/mirae/gongsi/corpus/output_holding/markdown"\
      "/Users/chanuyoung/Documents/2026/summer_intership/contest/mirae/gongsi/corpus/output_holding/chunks" \
      --min 800 --max 1000 --manifest "/Users/chanuyoung/Documents/2026/summer_intership/contest/mirae/gongsi/corpus/manifest.jsonl"
   
  batch 는 입력 디렉터리의 각 *.md 파일마다
      {out_dir}/{stem}.chunks.jsonl
  을 생성한다 (JSON Lines, 1행 = 청크 1개, 임베딩 API에 바로 사용 가능).

의존성: 표준 라이브러리만 사용.
---------------------------------------------------------------------
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

# =====================================================================
# 0. 파라미터 기본값
# =====================================================================

DEFAULT_MIN_CHARS = 800
DEFAULT_MAX_CHARS = 1000
# 문단이 min_chars 를 채우지 못한 채로 다음 블록이 안 들어가는 경우,
# 약간의 초과(overflow)를 허용해서 어색하게 짧은 청크가 남는 것을 줄인다.
OVERFLOW_RATIO = 1.15
# 표가 혼자서 이 크기를 넘으면 행 단위로 쪼갠다.
TABLE_HARD_MAX_RATIO = 1.3

_TABLE_LINE_RE = re.compile(r'^\s*\|.*\|\s*$')
_TABLE_SEP_RE = re.compile(r'^\s*\|[\s:\-|]+\|\s*$')  # |---|---| 형태 구분행
_HEADING_RE = re.compile(r'^(#{1,6})\s+(.*)$')
_LIST_ITEM_RE = re.compile(r'^\s*(?:[-*]\s+|\d+\.\s+)')
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?다요음함])\s+|\n')


# =====================================================================
# 1. 마크다운 -> 블록 파싱
# =====================================================================

def parse_markdown_blocks(md_text: str) -> list[dict]:
    """마크다운 텍스트를 의미 단위 블록 리스트로 분할한다.

    블록 타입: heading | table | blockquote | list | paragraph | hr

    각 블록: {'type', 'text', 'heading_path': list[str]}
    heading_path 는 해당 블록이 속한 시점까지의 헤딩 스택
    (예: ['주요사항보고서(자기주식 처분 결정)', '자기주식 처분 결정']).
    """
    lines = md_text.replace('\r\n', '\n').split('\n')
    blocks: list[dict] = []
    heading_stack: dict[int, str] = {}  # level -> title

    def current_path() -> list[str]:
        return [heading_stack[lv] for lv in sorted(heading_stack)]

    i, n = 0, len(lines)
    while i < n:
        line = lines[i]

        if not line.strip():
            i += 1
            continue

        # --- 헤딩 ---
        m = _HEADING_RE.match(line)
        if m:
            level, title = len(m.group(1)), m.group(2).strip()
            blocks.append({'type': 'heading', 'text': line.strip(),
                            'level': level, 'heading_path': current_path()})
            heading_stack[level] = title
            for lv in list(heading_stack):
                if lv > level:
                    del heading_stack[lv]
            i += 1
            continue

        # --- 표 (연속된 '|...|' 라인 묶음) ---
        if _TABLE_LINE_RE.match(line):
            j = i
            table_lines = []
            while j < n and _TABLE_LINE_RE.match(lines[j]):
                table_lines.append(lines[j])
                j += 1
            blocks.append({'type': 'table', 'text': '\n'.join(table_lines),
                            'heading_path': current_path()})
            i = j
            continue

        # --- 인용구(> ...) : 표 앞 안내문 등 ---
        if line.lstrip().startswith('>'):
            j = i
            quote_lines = []
            while j < n and lines[j].lstrip().startswith('>'):
                quote_lines.append(lines[j])
                j += 1
            blocks.append({'type': 'blockquote', 'text': '\n'.join(quote_lines),
                            'heading_path': current_path()})
            i = j
            continue

        # --- 리스트(연속된 -, 1. 항목, 빈 줄로만 구분) ---
        if _LIST_ITEM_RE.match(line):
            j = i
            list_lines = []
            while j < n and (_LIST_ITEM_RE.match(lines[j]) or
                              (lines[j].strip() and not lines[j].lstrip().startswith(('#', '|', '>')))):
                list_lines.append(lines[j])
                j += 1
            blocks.append({'type': 'list', 'text': '\n'.join(list_lines),
                            'heading_path': current_path()})
            i = j
            continue

        # --- 일반 문단 (빈 줄 전까지 이어붙임) ---
        j = i
        para_lines = []
        while j < n and lines[j].strip() and not (
                _HEADING_RE.match(lines[j]) or _TABLE_LINE_RE.match(lines[j])
                or lines[j].lstrip().startswith('>') or _LIST_ITEM_RE.match(lines[j])):
            para_lines.append(lines[j])
            j += 1
        blocks.append({'type': 'paragraph', 'text': '\n'.join(para_lines),
                        'heading_path': current_path()})
        i = j

    return blocks


# =====================================================================
# 2. 오버사이즈 블록 분할 (표 / 일반 텍스트)
# =====================================================================

def _split_table_block(block: dict, max_chars: int) -> list[dict]:
    """표 블록이 max_chars 를 크게 초과할 때, 헤더(+구분행)를 반복하며
    '행' 단위로만 잘라 여러 표 블록으로 나눈다. 행 내부는 절대 자르지 않는다."""
    lines = block['text'].split('\n')
    if len(lines) <= 2:
        return [block]  # 헤더뿐이거나 행이 거의 없음 -> 더 못 쪼갬

    header_lines = [lines[0]]
    body_start = 1
    if _TABLE_SEP_RE.match(lines[1]):
        header_lines.append(lines[1])
        body_start = 2
    body_lines = lines[body_start:]

    header_text = '\n'.join(header_lines)
    header_len = len(header_text) + 1

    parts: list[list[str]] = []
    cur: list[str] = []
    cur_len = header_len
    for row in body_lines:
        row_len = len(row) + 1
        if cur and cur_len + row_len > max_chars:
            parts.append(cur)
            cur, cur_len = [], header_len
        cur.append(row)
        cur_len += row_len
    if cur:
        parts.append(cur)

    out = []
    total = len(parts)
    for idx, rows in enumerate(parts):
        text = '\n'.join(header_lines + rows)
        out.append({'type': 'table', 'text': text,
                     'heading_path': block['heading_path'],
                     'table_part': f'{idx + 1}/{total}' if total > 1 else None})
    return out


def _split_text_block(block: dict, max_chars: int) -> list[dict]:
    """표가 아닌 블록(문단 등)이 max_chars 를 넘을 때 문장/개행 경계로 분할."""
    text = block['text']
    if len(text) <= max_chars:
        return [block]
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if not sentences:
        sentences = [text[k:k + max_chars] for k in range(0, len(text), max_chars)]

    out_texts, cur = [], ''
    for s in sentences:
        cand = (cur + ' ' + s).strip() if cur else s
        if cur and len(cand) > max_chars:
            out_texts.append(cur)
            cur = s
        else:
            cur = cand
        # 문장 하나 자체가 너무 길면 강제로 자름 (극단적 경우 대비)
        while len(cur) > max_chars:
            out_texts.append(cur[:max_chars])
            cur = cur[max_chars:]
    if cur:
        out_texts.append(cur)

    return [{'type': block['type'], 'text': t, 'heading_path': block['heading_path']}
            for t in out_texts]


# =====================================================================
# 3. 블록 -> 청크
# =====================================================================

# 수정 후 (embedding_text 필드 제거 또는 간소화)
def _make_chunk(buf_blocks: list[dict], idx: int, doc_id: Optional[str]) -> dict:
    text = '\n\n'.join(b['text'] for b in buf_blocks)
    content_blocks = [b for b in buf_blocks if b['type'] != 'heading']
    ref = content_blocks[-1] if content_blocks else buf_blocks[-1]
    heading_path = ' > '.join(ref['heading_path'])

    # 💡 굳이 전체 텍스트를 또 복사해 넣지 않고, 나중에 검색할 때 필요한 
    # 'heading_path'와 'text'만 각각 저장하거나 embedding_text를 아예 빼버립니다!
    # (embed_chunks.py가 없어도 text를 알아서 읽어 쓰므로 과감히 제거해도 됩니다)
    
    return {
        'chunk_id': f'{doc_id}_{idx}' if doc_id else str(idx),
        'chunk_index': idx,
        'heading_path': heading_path,
        'has_table': any(b['type'] == 'table' for b in buf_blocks),
        'char_count': len(text),
        'text': text,
        # 'embedding_text' 필드 삭제 완료! 용량이 절반으로 줍니다.
    }


def chunk_blocks(blocks: list[dict], min_chars: int = DEFAULT_MIN_CHARS,
                  max_chars: int = DEFAULT_MAX_CHARS,
                  doc_id: Optional[str] = None) -> list[dict]:
    """파싱된 블록 리스트를 min_chars~max_chars 구간의 청크들로 합친다.
    표는 절대 행 중간에서 자르지 않는다 (필요 시 _split_table_block 로 사전 분할).
    doc_id 를 주면 각 청크에 전역 고유 chunk_id(f'{doc_id}_{chunk_index}')를 부여한다."""
    overflow_max = int(max_chars * OVERFLOW_RATIO)
    table_hard_max = int(max_chars * TABLE_HARD_MAX_RATIO)

    # 3-1) 오버사이즈 블록 사전 분할 (표 / 일반 텍스트 각각 별도 규칙)
    expanded: list[dict] = []
    for b in blocks:
        if b['type'] == 'table' and len(b['text']) > table_hard_max:
            expanded.extend(_split_table_block(b, max_chars))
        elif b['type'] != 'table' and len(b['text']) > overflow_max:
            expanded.extend(_split_text_block(b, max_chars))
        else:
            expanded.append(b)

    chunks: list[dict] = []
    buf: list[dict] = []
    buf_len = 0

    def sep_len() -> int:
        return 2 if buf else 0  # '\n\n'

    def flush():
        nonlocal buf, buf_len
        if buf:
            chunks.append(_make_chunk(buf, len(chunks), doc_id))
        buf, buf_len = [], 0

    def pull_trailing_heading() -> Optional[dict]:
        """버퍼 맨 끝이 헤딩뿐이면 꺼내서 다음 청크로 이월시킨다
        (헤딩이 청크 끝에 고립되는 것을 방지)."""
        nonlocal buf, buf_len
        if buf and buf[-1]['type'] == 'heading':
            h = buf.pop()
            buf_len -= len(h['text']) + (2 if buf else 0)
            return h
        return None

    for block in expanded:
        blen = len(block['text'])

        if block['type'] == 'table':
            # 표는 통째로 들어갈 수 있는 청크에만 배치한다.
            if buf and buf_len + sep_len() + blen <= max_chars:
                buf.append(block)
                buf_len += sep_len() + blen
                continue
            # 안 들어가면 지금까지 버퍼를 닫고, 표만 단독 청크로.
            carry = pull_trailing_heading()
            flush()
            new_buf = [carry] if carry else []
            new_len = (len(carry['text']) if carry else 0)
            new_buf.append(block)
            new_len += (2 if carry else 0) + blen
            buf, buf_len = new_buf, new_len
            if buf_len > table_hard_max:
                # 그래도 여전히 크면 (carry 헤딩 포함) 바로 닫는다.
                flush()
            continue

        # 표가 아닌 블록
        cand_len = buf_len + sep_len() + blen
        if not buf:
            buf, buf_len = [block], blen
        elif cand_len <= max_chars:
            buf.append(block)
            buf_len = cand_len
        elif buf_len < min_chars and cand_len <= overflow_max:
            # 아직 min_chars 를 못 채웠으면 약간의 초과를 허용해서라도 채운다.
            buf.append(block)
            buf_len = cand_len
        else:
            carry = pull_trailing_heading()
            flush()
            if carry:
                buf, buf_len = [carry, block], len(carry['text']) + 2 + blen
            else:
                buf, buf_len = [block], blen

    flush()
    return chunks


# =====================================================================
# 4. 파일/디렉터리 처리
# =====================================================================

def chunk_markdown_text(md_text: str, min_chars: int = DEFAULT_MIN_CHARS,
                         max_chars: int = DEFAULT_MAX_CHARS,
                         doc_id: Optional[str] = None) -> list[dict]:
    """마크다운 문자열 하나를 청크 리스트(dict)로 변환하는 최상위 함수."""
    blocks = parse_markdown_blocks(md_text)
    return chunk_blocks(blocks, min_chars=min_chars, max_chars=max_chars, doc_id=doc_id)


def chunk_markdown_file(path, min_chars: int = DEFAULT_MIN_CHARS,
                         max_chars: int = DEFAULT_MAX_CHARS,
                         doc_id: Optional[str] = None) -> list[dict]:
    text = Path(path).read_text(encoding='utf-8')
    doc_id = doc_id if doc_id is not None else Path(path).stem
    return chunk_markdown_text(text, min_chars=min_chars, max_chars=max_chars, doc_id=doc_id)


# manifest.jsonl 에서 청크 레코드에 합쳐 넣을 문서 메타데이터 필드
_MANIFEST_JOIN_FIELDS = (
    'corp_name', 'listed_name', 'stock_code', 'industry', 'sector',
    'doc_group', 'doc_subtype', 'report_nm', 'rcept_dt', 'is_correction',
)


def _load_manifest_lookup(manifest_path) -> dict[str, dict]:
    """manifest.jsonl 을 doc_id -> {_MANIFEST_JOIN_FIELDS 값} 딕셔너리로 로드."""
    lookup: dict[str, dict] = {}
    with open(manifest_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            doc_id = rec.get('doc_id')
            if doc_id:
                lookup[doc_id] = {k: rec.get(k) for k in _MANIFEST_JOIN_FIELDS}
    return lookup


def batch_chunk(in_dir, out_dir, min_chars: int = DEFAULT_MIN_CHARS,
                 max_chars: int = DEFAULT_MAX_CHARS, pattern: str = '*.md',
                 manifest_path=None):
    """디렉터리 내 모든 마크다운 파일을 청킹해 {out_dir}/{doc_id}.chunks.jsonl 로 저장한다.
    manifest_path 를 주면 doc_id 기준으로 manifest.jsonl 의 문서 메타데이터
    (corp_name/sector/doc_group/rcept_dt 등)를 각 청크 레코드에 합쳐 넣는다."""
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_files = sorted(in_dir.glob(pattern))
    if not md_files:
        print(f'[WARN] {in_dir} 안에 {pattern} 파일이 없습니다.', file=sys.stderr)
        return 0, 0

    manifest_lookup = _load_manifest_lookup(manifest_path) if manifest_path else {}
    n_meta_miss = 0

    n_ok, n_fail = 0, 0
    for md_path in md_files:
        doc_id = md_path.stem
        try:
            chunks = chunk_markdown_file(md_path, min_chars=min_chars, max_chars=max_chars,
                                          doc_id=doc_id)
            meta = manifest_lookup.get(doc_id)
            if manifest_lookup and meta is None:
                n_meta_miss += 1
            out_path = out_dir / f'{doc_id}.chunks.jsonl'
            with open(out_path, 'w', encoding='utf-8') as f:
                for ch in chunks:
                    rec = {'doc_id': doc_id, **(meta or {}), **ch}
                    f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            n_ok += 1
        except Exception as e:
            n_fail += 1
            print(f'[FAIL] {doc_id}: {e}', file=sys.stderr)
    if manifest_lookup and n_meta_miss:
        print(f'[WARN] manifest 에서 메타데이터를 못 찾은 문서 {n_meta_miss}건 '
              f'(doc_id 불일치 가능성)', file=sys.stderr)
    print(f'batch_chunk 완료: 성공 {n_ok}건 / 실패 {n_fail}건 -> {out_dir}', file=sys.stderr)
    return n_ok, n_fail


# =====================================================================
# CLI
# =====================================================================

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='마크다운 문서 800~1000자 청킹 (표 안 깨짐)')
    sub = ap.add_subparsers(dest='cmd', required=True)

    p1 = sub.add_parser('one', help='단일 .md 파일 청킹 (콘솔 출력)')
    p1.add_argument('md_path')
    p1.add_argument('--min', type=int, default=DEFAULT_MIN_CHARS)
    p1.add_argument('--max', type=int, default=DEFAULT_MAX_CHARS)

    p2 = sub.add_parser('batch', help='디렉터리 내 모든 .md 파일 일괄 청킹')
    p2.add_argument('in_dir')
    p2.add_argument('out_dir')
    p2.add_argument('--min', type=int, default=DEFAULT_MIN_CHARS)
    p2.add_argument('--max', type=int, default=DEFAULT_MAX_CHARS)
    p2.add_argument('--pattern', default='*.md')
    p2.add_argument('--manifest', default=None,
                     help='manifest.jsonl 경로. 주면 corp_name/sector/doc_group 등을 청크에 조인.')

    args = ap.parse_args()

    if args.cmd == 'one':
        chunks = chunk_markdown_file(args.md_path, min_chars=args.min, max_chars=args.max)
        print(f'총 {len(chunks)}개 청크 생성 (목표 {args.min}~{args.max}자)\n')
        for ch in chunks:
            print(f"--- {ch['chunk_id']} | {ch['char_count']}자 "
                  f"| table={ch['has_table']} | {ch['heading_path']} ---")
            print(ch['text'])
            print()
    elif args.cmd == 'batch':
        batch_chunk(args.in_dir, args.out_dir, min_chars=args.min, max_chars=args.max,
                    pattern=args.pattern, manifest_path=args.manifest)
