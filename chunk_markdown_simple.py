# -*- coding: utf-8 -*-
"""
chunk_markdown_simple.py
=====================================================================
DART 공시 마크다운을 "헤더 기준 섹션 분할 + 재귀적 문자 분할" 방식으로
청킹한다. (표를 행 단위로 다루는 chunk_markdown.py 대신, 훨씬 단순하고
예측 가능한 방식 — 실측 기준 원본 대비 약 1.3~1.4배 크기로만 증가한다.)

핵심 원칙
---------------------------------------------------------------------
1. 먼저 헤딩(#, ##, ###)을 기준으로 문서를 섹션으로 나눈다.
2. 각 섹션 본문을 chunk_size(기본 1000자) 이하가 되도록 재귀적으로
   분할한다. 구분자 우선순위: 빈 줄 -> 줄바꿈 -> 마침표 -> 공백 -> 강제분할.
3. 문맥 유지를 위해 overlap(기본 150자)만큼 앞 청크의 꼬리를 다음
   청크 앞에 이어 붙인다.
4. 헤딩 경로(heading_path)는 텍스트 안에 박아 넣지 않고 **메타데이터
   필드로만** 저장한다 (청크 본문에 매번 프리픽스를 반복해서 넣으면
   그 자체로 용량이 불어나기 때문 — 실측 시 전체 증가분의 약 1/4을 차지).
5. 표(|...|)가 포함된 청크는 has_table=True로만 표시한다. (표를 행
   단위로 안전하게 다루는 정교한 처리는 하지 않음 — 대신 훨씬 단순하고
   오동작 가능성이 적다.)

출력: chunk_markdown.py 와 동일한 필드명(chunk_id, chunk_index,
heading_path, has_table, char_count, text)을 사용하므로
embed_chunks.py 를 그대로 이어서 쓸 수 있다.

사용법
---------------------------------------------------------------------
  단일 파일 (콘솔에 통계 출력):
    python3 chunk_markdown_simple.py one "periodic_20230512000710.md" \
        --chunk-size 1000 --overlap 150

  디렉터리 일괄 처리:
    python3 chunk_markdown_simple.py batch \
        "/path/to/markdown" "/path/to/chunks" \
        --chunk-size 1000 --overlap 150 \
        --manifest "/path/to/manifest.jsonl"

  batch 는 입력 디렉터리의 각 *.md 파일마다
      {out_dir}/{stem}.chunks.jsonl
  을 생성한다.

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

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_OVERLAP = 150
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]  # 우선순위: 문단 > 줄 > 문장 > 단어 > 강제분할

_HEADING_RE = re.compile(r'^(#{1,6})\s+(.*)$', re.MULTILINE)


# =====================================================================
# 1. 헤더 기준 섹션 분할
# =====================================================================

def split_into_sections(md_text: str) -> list[dict]:
    """마크다운을 헤딩 기준 섹션으로 나눈다.
    각 섹션: {'heading_path': list[str], 'text': str}
    """
    matches = list(_HEADING_RE.finditer(md_text))
    sections: list[dict] = []
    stack: dict[int, str] = {}

    if matches:
        preamble = md_text[:matches[0].start()].strip('\n')
        if preamble:
            sections.append({'heading_path': [], 'text': preamble})
    else:
        return [{'heading_path': [], 'text': md_text}]

    for i, m in enumerate(matches):
        level, title = len(m.group(1)), m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md_text)
        body = md_text[start:end].strip('\n')

        stack[level] = title
        for lv in list(stack):
            if lv > level:
                del stack[lv]
        heading_path = [stack[lv] for lv in sorted(stack)]

        if body:
            sections.append({'heading_path': heading_path, 'text': body})

    return sections


# =====================================================================
# 2. 재귀적 문자 분할 (오버랩 포함)
# =====================================================================

def recursive_split(text: str, chunk_size: int, overlap: int,
                     seps: list[str] = SEPARATORS) -> list[str]:
    """text 를 chunk_size 이하 조각으로 재귀 분할. overlap 만큼 앞 조각의
    꼬리를 다음 조각 앞에 이어 붙여 문맥이 끊기지 않게 한다."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    sep = seps[0] if seps else ""
    remaining = seps[1:] if len(seps) > 1 else []

    if sep == "":
        step = max(chunk_size - overlap, 1)
        return [text[i:i + chunk_size] for i in range(0, len(text), step)]

    parts = [p for p in text.split(sep) if p != ""]
    chunks: list[str] = []
    current = ""

    for part in parts:
        candidate = f"{current}{sep}{part}" if current else part
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)
            tail = current[-overlap:] if overlap < len(current) else current
            current = f"{tail}{sep}{part}"
        else:
            current = part

        if len(current) > chunk_size:
            sub = recursive_split(current, chunk_size, overlap, remaining)
            if sub:
                chunks.extend(sub[:-1])
                current = sub[-1]
            else:
                current = ""

    if current.strip():
        chunks.append(current)
    return chunks


# =====================================================================
# 3. 마크다운 -> 청크
# =====================================================================

def chunk_markdown_text(md_text: str, chunk_size: int = DEFAULT_CHUNK_SIZE,
                         overlap: int = DEFAULT_OVERLAP,
                         doc_id: Optional[str] = None) -> list[dict]:
    sections = split_into_sections(md_text)
    chunks: list[dict] = []

    for section in sections:
        pieces = recursive_split(section['text'], chunk_size, overlap)
        for piece in pieces:
            idx = len(chunks)
            chunks.append({
                'chunk_id': f'{doc_id}_{idx}' if doc_id else str(idx),
                'chunk_index': idx,
                'heading_path': ' > '.join(section['heading_path']),
                'has_table': '|' in piece,
                'char_count': len(piece),
                'text': piece,
            })
    return chunks


def chunk_markdown_file(path, chunk_size: int = DEFAULT_CHUNK_SIZE,
                         overlap: int = DEFAULT_OVERLAP,
                         doc_id: Optional[str] = None) -> list[dict]:
    text = Path(path).read_text(encoding='utf-8')
    doc_id = doc_id if doc_id is not None else Path(path).stem
    return chunk_markdown_text(text, chunk_size=chunk_size, overlap=overlap, doc_id=doc_id)


# =====================================================================
# 4. manifest 조인 + 배치 처리
# =====================================================================

_MANIFEST_JOIN_FIELDS = (
    'corp_name', 'listed_name', 'stock_code', 'industry', 'sector',
    'doc_group', 'doc_subtype', 'report_nm', 'rcept_dt', 'is_correction',
)


def _load_manifest_lookup(manifest_path) -> dict[str, dict]:
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


def batch_chunk(in_dir, out_dir, chunk_size: int = DEFAULT_CHUNK_SIZE,
                 overlap: int = DEFAULT_OVERLAP, pattern: str = '*.md',
                 manifest_path=None):
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
            chunks = chunk_markdown_file(md_path, chunk_size=chunk_size,
                                          overlap=overlap, doc_id=doc_id)
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
    ap = argparse.ArgumentParser(
        description='마크다운 문서를 헤더분할+재귀적문자분할로 청킹 (표는 행단위로 다루지 않음)')
    sub = ap.add_subparsers(dest='cmd', required=True)

    p1 = sub.add_parser('one', help='단일 .md 파일 청킹 (콘솔에 통계+미리보기 출력)')
    p1.add_argument('md_path')
    p1.add_argument('--chunk-size', type=int, default=DEFAULT_CHUNK_SIZE)
    p1.add_argument('--overlap', type=int, default=DEFAULT_OVERLAP)

    p2 = sub.add_parser('batch', help='디렉터리 내 모든 .md 파일 일괄 청킹')
    p2.add_argument('in_dir')
    p2.add_argument('out_dir')
    p2.add_argument('--chunk-size', type=int, default=DEFAULT_CHUNK_SIZE)
    p2.add_argument('--overlap', type=int, default=DEFAULT_OVERLAP)
    p2.add_argument('--pattern', default='*.md')
    p2.add_argument('--manifest', default=None,
                     help='manifest.jsonl 경로. 주면 corp_name/sector/doc_group 등을 청크에 조인.')

    args = ap.parse_args()

    if args.cmd == 'one':
        orig_text = Path(args.md_path).read_text(encoding='utf-8')
        chunks = chunk_markdown_file(args.md_path, chunk_size=args.chunk_size,
                                      overlap=args.overlap)
        total_chars = sum(c['char_count'] for c in chunks)
        print(f"원본: {len(orig_text):,}자")
        print(f"청크: {len(chunks)}개, 합계 {total_chars:,}자 "
              f"({total_chars / len(orig_text):.2f}배)\n")
        for ch in chunks:
            print(f"--- {ch['chunk_id']} | {ch['char_count']}자 "
                  f"| table={ch['has_table']} | {ch['heading_path']} ---")
            print(ch['text'])
            print()
    elif args.cmd == 'batch':
        batch_chunk(args.in_dir, args.out_dir, chunk_size=args.chunk_size,
                    overlap=args.overlap, pattern=args.pattern,
                    manifest_path=args.manifest)
