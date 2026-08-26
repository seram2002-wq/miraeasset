# -*- coding: utf-8 -*-
"""
search_faiss.py (파트 C 연동 완성본)
=====================================================================
[코드 설명]
  - 사용자 질문을 Clova Studio Embedding API로 벡터화한 뒤,
    FAISS 인덱스에서 관련 공시 청크를 초고속(Top-K) 검색합니다.
  - [파트 C 정정공시 최신본 우선 로직]:
    * 동일한 lineage_id(동일 사건/보고서 계보) 내에서 여러 청크가 잡힐 경우,
      기본적으로 is_latest=True 인 최종 정정본 청크를 최우선으로 선별합니다.
    * --latest-only 옵션(기본 활성화)으로 구버전(정정 전) 문서를 자동 필터링하여 환각을 방지합니다.
    * --distinct-doc 옵션으로 동일 계보당 최상위 1개 청크만 노출 가능합니다.
"""

import argparse
import json
import os
import pickle
import sys
import uuid
from pathlib import Path
import numpy as np
import requests

try:
    import faiss
except ImportError:
    faiss = None

EMBEDDING_URL = "https://clovastudio.stream.ntruss.com/testapp/v1/api-tools/embedding/v2"


def embed_query(query: str, api_key: str) -> list[float]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "X-NCP-CLOVASTUDIO-REQUEST-ID": str(uuid.uuid4()),
        "Content-Type": "application/json",
    }
    payload = {"text": query}
    try:
        resp = requests.post(EMBEDDING_URL, headers=headers, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        raise RuntimeError(f"Clova Embedding API 호출 실패: {e}") from e

    status_code = data.get("status", {}).get("code")
    if status_code != "20000":
        raise RuntimeError(f"Clova API 오류 응답: {data.get('status')}")
    return data["result"]["embedding"]


class FaissSearcher:
    def __init__(self, index_dir: str | Path):
        if faiss is None:
            raise ImportError("faiss-cpu 라이브러리가 필요합니다 (pip install faiss-cpu).")

        index_dir = Path(index_dir)
        index_path = index_dir / "index.faiss"
        meta_path = index_dir / "metadata.pkl"

        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"인덱스 또는 메타데이터를 찾을 수 없습니다: {index_dir}")

        print(f"[LOAD] FAISS 인덱스 로딩: {index_path}")
        self.index = faiss.read_index(str(index_path))

        print(f"[LOAD] 메타데이터 로딩: {meta_path}")
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)

        print(f"[READY] 검색기 준비 완료! (총 {len(self.metadata):,}개 청크 등록됨)")

    def search(
        self,
        query: str,
        api_key: str,
        top_k: int = 5,
        corp_name: str | None = None,
        source_doc_type: str | None = None,
        doc_group: str | None = None,
        latest_only: bool = True,
        distinct_doc: bool = False,
        fetch_k: int = 50,
    ) -> list[dict]:
        q_vec = embed_query(query, api_key)
        q_arr = np.array([q_vec], dtype=np.float32)
        faiss.normalize_L2(q_arr)

        total_chunks = self.index.ntotal
        has_filter = bool(corp_name or source_doc_type or doc_group or latest_only or distinct_doc)
        current_fetch_k = max(top_k * 5, fetch_k) if has_filter else top_k
        current_fetch_k = min(current_fetch_k, total_chunks)

        results = []
        max_attempts = 3

        for attempt in range(max_attempts):
            distances, indices = self.index.search(q_arr, current_fetch_k)

            seen_chunk_ids = set()
            seen_lineages = set()
            results = []

            for score, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(self.metadata):
                    continue
                meta = dict(self.metadata[idx])
                meta["score"] = float(score)

                if corp_name and meta.get("corp_name") != corp_name:
                    continue
                if source_doc_type and meta.get("source_doc_type") != source_doc_type:
                    continue
                if doc_group and meta.get("doc_group") != doc_group:
                    continue

                if latest_only and not meta.get("is_latest", True):
                    continue

                chk_id = meta.get("chunk_id")
                if chk_id and chk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(chk_id)

                lin_id = meta.get("lineage_id")
                if distinct_doc and lin_id:
                    if lin_id in seen_lineages:
                        continue
                    seen_lineages.add(lin_id)

                results.append(meta)
                if len(results) >= top_k:
                    break

            if len(results) >= top_k or current_fetch_k >= total_chunks:
                break

            current_fetch_k = min(current_fetch_k * 4, total_chunks)

        return results


def main():
    ap = argparse.ArgumentParser(description="FAISS 공시 검색기 (Retrieval & Lineage)")
    ap.add_argument("--index-dir", default="faiss_index", help="FAISS 인덱스 디렉터리")
    ap.add_argument("--query", required=True, help="검색할 질문 문장")
    ap.add_argument("--top-k", type=int, default=5, help="반환할 상위 결과 개수")
    ap.add_argument("--corp-name", default=None, help="기업명 필터 (예: 삼성전자)")
    ap.add_argument("--doc-type", default=None, help="서식 필터 (예: 거래소공시, 정기공시, 주요사항보고서, 지분공시)")
    ap.add_argument("--doc-group", default=None, help="영문 서식 그룹 필터 (periodic, major, exchange, holding)")
    ap.add_argument("--distinct-doc", action="store_true", help="동일 문서(계보)당 최상위 청크 1개만 반환")
    ap.add_argument("--include-old", action="store_true", help="정정 전 구버전 공시도 결과에 포함")
    ap.add_argument("--api-key", default=None, help="CLOVA API Key")
    args = ap.parse_args()

    api_key = args.api_key or os.environ.get("CLOVA_API_KEY")
    if not api_key:
        print("[ERROR] --api-key 를 주거나 환경변수 CLOVA_API_KEY 를 설정하세요.", file=sys.stderr)
        sys.exit(1)

    searcher = FaissSearcher(args.index_dir)
    results = searcher.search(
        query=args.query,
        api_key=api_key,
        top_k=args.top_k,
        corp_name=args.corp_name,
        source_doc_type=args.doc_type,
        doc_group=args.doc_group,
        latest_only=not args.include_old,
        distinct_doc=args.distinct_doc,
    )

    print()
    print(f"===== 검색 결과 (질문: {args.query}) =====")
    if not results:
        print("조건에 맞는 검색 결과가 없습니다.")
        return

    for rank, item in enumerate(results, start=1):
        latest_badge = "★[최종 최신본]" if item.get("is_latest", True) else "[구버전/정정전]"
        print(f"\n[결과 {rank}] 유사도 점수: {item.get('score', 0.0):.4f} {latest_badge}")
        print(f"- 기업명: {item.get('corp_name')} ({item.get('stock_code')}) | 서식: {item.get('source_doc_type')} ({item.get('doc_group')})")
        print(f"- 보고서: {item.get('report_nm')} (접수일: {item.get('rcept_dt')})")
        print(f"- 계보ID: {item.get('lineage_id')} (버전: {item.get('version_order')}/{item.get('total_versions')})")
        print(f"- 섹션  : {item.get('section') or item.get('section_path')}")
        print("--- [본문 내용] ---")
        text = item.get("text", "")
        print(text[:300] + ("..." if len(text) > 300 else ""))
        print("-" * 50)


if __name__ == "__main__":
    main()

