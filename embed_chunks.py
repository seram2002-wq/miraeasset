# -*- coding: utf-8 -*-
"""
embed_chunks.py
=====================================================================
chunk_markdown.py 가 만든 {doc_id}.chunks.jsonl 파일들을 읽어서,
각 청크의 `embedding_text` 필드를 CLOVA Studio 임베딩v2 API로 벡터화하고
{chunk_id, embedding, ...메타데이터} 형태로 하나의 결과 파일에 저장한다.

특징
---------------------------------------------------------------------
- 순차 호출 + 재시도(지수 백오프) : rate limit / 일시적 오류에 안전
- 이어하기(resume) 지원 : 이미 처리된 chunk_id는 건너뜀
  (중간에 끊겨도 다시 실행하면 이어서 진행됨)
- 진행 상황 주기적으로 콘솔 출력

사용법
---------------------------------------------------------------------
  export CLOVA_API_KEY="nv-xxxxxxxx"

  python3 /Users/chanuyoung/embed_chunks.py \
      --chunks-dir "/Users/chanuyoung/Documents/2026/summer_intership/contest/mirae/gongsi/corpus/output_holding/chunks" \
      --out "/Users/chanuyoung/Documents/2026/summer_intership/contest/mirae/gongsi/corpus/output_holding/embeddings.jsonl" \
      --pattern "*.chunks.jsonl"


      
출력 (embeddings.jsonl, 1행 = 청크 1개)
---------------------------------------------------------------------
  {"chunk_id": "...", "doc_id": "...", "corp_name": "...", "sector": "...",
   "doc_group": "...", "rcept_dt": "...", "embedding": [0.01, -0.23, ...]}

의존성: requests (pip install requests --break-system-packages)
---------------------------------------------------------------------
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import requests

EMBEDDING_URL = "https://clovastudio.stream.ntruss.com/v1/api-tools/embedding/v2"

# 청크 레코드에서 결과 파일에 함께 저장할 메타데이터 필드
# (chunk_markdown.py 의 batch_chunk() 출력 필드 기준)
_KEEP_META_FIELDS = (
    "chunk_id", "doc_id", "corp_name", "listed_name", "stock_code",
    "industry", "sector", "doc_group", "doc_subtype", "report_nm",
    "rcept_dt", "is_correction", "heading_path", "has_table", "char_count",
)


# =====================================================================
# 1. API 호출 (재시도 포함)
# =====================================================================

def embed_text(text: str, api_key: str, max_retries: int = 5,
                timeout: float = 15.0) -> list[float]:
    """텍스트 하나를 임베딩 벡터로 변환한다. 실패 시 지수 백오프로 재시도."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "X-NCP-CLOVASTUDIO-REQUEST-ID": str(uuid.uuid4()),
        "Content-Type": "application/json",
    }
    payload = {"text": text}

    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(EMBEDDING_URL, headers=headers,
                                  json=payload, timeout=timeout)
            # rate limit / 서버 오류는 재시도, 그 외 4xx는 즉시 실패
            if resp.status_code == 429 or resp.status_code >= 500:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            resp.raise_for_status()
            data = resp.json()
            status_code = data.get("status", {}).get("code")
            if status_code != "20000":
                raise RuntimeError(f"API 오류: {data.get('status')}")
            return data["result"]["embedding"]
        except Exception as e:  # noqa: BLE001
            last_err = e
            wait = min(2 ** attempt, 30)
            print(f"  [재시도 {attempt + 1}/{max_retries}] {e} -> {wait}초 대기",
                  file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"임베딩 실패 (재시도 {max_retries}회 초과): {last_err}")


# =====================================================================
# 2. 이어하기(resume) 지원
# =====================================================================

def _load_done_ids(out_path: Path) -> set[str]:
    """이미 결과 파일에 기록된 chunk_id 집합을 로드한다 (이어하기용)."""
    done: set[str] = set()
    if not out_path.exists():
        return done
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if "chunk_id" in rec:
                    done.add(rec["chunk_id"])
            except json.JSONDecodeError:
                continue
    return done


def _iter_chunk_records(chunks_dir: Path, pattern: str):
    """chunks_dir 내 모든 *.chunks.jsonl 파일을 순회하며 청크 레코드를 yield."""
    files = sorted(chunks_dir.glob(pattern))
    if not files:
        print(f"[WARN] {chunks_dir} 안에 {pattern} 파일이 없습니다.", file=sys.stderr)
    for path in files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


# =====================================================================
# 3. 메인 파이프라인
# =====================================================================

def run(chunks_dir: str, out_path: str, api_key: str, pattern: str = "*.chunks.jsonl",
        sleep_between: float = 0.05, progress_every: int = 100) -> None:
    chunks_dir_p = Path(chunks_dir)
    out_path_p = Path(out_path)
    out_path_p.parent.mkdir(parents=True, exist_ok=True)

    done_ids = _load_done_ids(out_path_p)
    if done_ids:
        print(f"[RESUME] 이미 처리된 청크 {len(done_ids)}건, 이어서 진행합니다.",
              file=sys.stderr)

    n_total = n_done = n_skipped = n_failed = 0
    t0 = time.time()

    # append 모드로 열어서 이어하기가 자연스럽게 되도록 함
    with open(out_path_p, "a", encoding="utf-8") as out_f:
        for rec in _iter_chunk_records(chunks_dir_p, pattern):
            n_total += 1
            chunk_id = rec.get("chunk_id")
            if not chunk_id:
                print(f"[SKIP] chunk_id 없는 레코드 건너뜀: {rec.get('doc_id')}",
                      file=sys.stderr)
                n_skipped += 1
                continue
            if chunk_id in done_ids:
                n_skipped += 1
                continue

            text = rec.get("embedding_text") or rec.get("text")
            if not text:
                print(f"[SKIP] 임베딩할 텍스트 없음: {chunk_id}", file=sys.stderr)
                n_skipped += 1
                continue

            try:
                vector = embed_text(text, api_key)
            except Exception as e:  # noqa: BLE001
                print(f"[FAIL] {chunk_id}: {e}", file=sys.stderr)
                n_failed += 1
                continue

            out_rec = {k: rec.get(k) for k in _KEEP_META_FIELDS if k in rec}
            out_rec["embedding"] = vector
            out_f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            out_f.flush()  # 중간에 끊겨도 유실 최소화
            n_done += 1

            if n_done % progress_every == 0:
                elapsed = time.time() - t0
                rate = n_done / elapsed if elapsed > 0 else 0
                print(f"[진행] {n_done}건 완료 (실패 {n_failed}, 스킵 {n_skipped}) "
                      f"- {rate:.1f}건/초", file=sys.stderr)

            if sleep_between > 0:
                time.sleep(sleep_between)

    elapsed = time.time() - t0
    print(f"\n완료: 총 {n_total}건 중 성공 {n_done} / 스킵(이미완료) {n_skipped} "
          f"/ 실패 {n_failed} ({elapsed:.1f}초) -> {out_path_p}", file=sys.stderr)
    if n_failed:
        print("[안내] 실패 건은 스크립트를 그대로 다시 실행하면 "
              "이어하기 로직에 의해 재시도됩니다 (성공분은 건너뜁니다).",
              file=sys.stderr)


# =====================================================================
# CLI
# =====================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="chunk_markdown.py 출력(.chunks.jsonl)을 CLOVA 임베딩v2로 벡터화")
    ap.add_argument("--chunks-dir", required=True, help="청크 jsonl들이 있는 디렉터리")
    ap.add_argument("--out", required=True, help="결과 저장 경로 (jsonl)")
    ap.add_argument("--pattern", default="*.chunks.jsonl")
    ap.add_argument("--api-key", default=None,
                     help="CLOVA API Key. 미지정 시 환경변수 CLOVA_API_KEY 사용")
    ap.add_argument("--sleep", type=float, default=0.05,
                     help="호출 사이 대기 시간(초), rate limit 방지용")
    args = ap.parse_args()

    api_key = args.api_key or os.environ.get("CLOVA_API_KEY")
    if not api_key:
        print("[ERROR] --api-key 를 주거나 환경변수 CLOVA_API_KEY 를 설정하세요.",
              file=sys.stderr)
        sys.exit(1)

    run(args.chunks_dir, args.out, api_key, pattern=args.pattern,
        sleep_between=args.sleep)
