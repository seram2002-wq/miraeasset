# -*- coding: utf-8 -*-
"""
search_faiss.py
=====================================================================
[코드 설명]
  - 사용자 자연어 질문을 Clova Studio Embedding API로 벡터화한 뒤,
    FAISS 인덱스(index.faiss)에서 가장 유사한 공시 본문 청크들을 초고속(Top-K)으로 검색(Retrieval)합니다.
  - 특정 기업명(--corp-name)이나 특정 서식(--doc-type) 조건 필터링을 지원합니다.

[필요 라이브러리 설치]
  pip install faiss-cpu requests numpy

[실행 방법 (CLI 터미널)]
  # 1. API 키 환경변수 설정
  $env:CLOVA_API_KEY="nv-xxxxxxxx" (PowerShell) 또는 export CLOVA_API_KEY="nv-xxxxxxxx" (Mac/Linux)

  # 2. 통합 질문 검색 (전체 서식 대상)
  python search_faiss.py --query "삼성전자 반도체 공급 계약 금액 얼마야?"

  # 3. 특정 기업 및 공시 서식 필터링 검색
  python search_faiss.py --query "유상증자 및 전환사채 발행 내역" --corp-name "고려아연" --doc-type "주요사항보고서" --top-k 3

[파이썬 코드 내에서 모듈로 임포트하여 사용할 때]
  from search_faiss import FaissSearcher

  searcher = FaissSearcher("faiss_index")
  results = searcher.search(
      query="최근 5% 이상 대량보유 보고자 및 변동 목적",
      api_key="nv-xxxx",
      corp_name="SK하이닉스",
      source_doc_type="지분공시",
      top_k=5
  )
  for r in results:
      print(r["score"], r["corp_name"], r["report_nm"], r["text"])

[주의사항]
  1. build_faiss_index.py 로 만들어진 index.faiss 와 metadata.pkl 이 존재하는 폴더(--index-dir)를 지정해야 합니다.
  2. 질문을 임베딩할 때 공시 문서를 임베딩했던 것과 동일한 Clova Studio Embedding 모델/API를 사용해야 정확한 검색이 이루어집니다.
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
    resp = requests.post(EMBEDDING_URL, headers=headers, json=payload, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    status_code = data.get("status", {}).get("code")
    if status_code != "20000":
        raise RuntimeError(f"Clova API 오류: {data.get('status')}")
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

        print(f"[LOAD] FAISS 인덱스 로딩 중: {index_path}")
        self.index = faiss.read_index(str(index_path))

        print(f"[LOAD] 메타데이터 로딩 중: {meta_path}")
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
        fetch_k: int = 50,
    ) -> list[dict]:
        # 1. 쿼리 임베딩
        q_vec = embed_query(query, api_key)
        q_arr = np.array([q_vec], dtype=np.float32)
        faiss.normalize_L2(q_arr)

        # 2. 필터링 조건을 위해 넉넉하게 fetch_k개 검색
        k_to_search = max(top_k * 5, fetch_k) if (corp_name or source_doc_type or doc_group) else top_k
        k_to_search = min(k_to_search, self.index.ntotal)

        distances, indices = self.index.search(q_arr, k_to_search)

        results = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = dict(self.metadata[idx])
            meta["score"] = float(score)

            # 필터링 조건 검사 (Post-filtering)
            if corp_name and meta.get("corp_name") != corp_name:
                continue
            if source_doc_type and meta.get("source_doc_type") != source_doc_type:
                continue
            if doc_group and meta.get("doc_group") != doc_group:
                continue

            results.append(meta)
            if len(results) >= top_k:
                break

        return results


def main():
    ap = argparse.ArgumentParser(description="FAISS 공시 검색기 (Retrieval)")
    ap.add_argument("--index-dir", default="faiss_index", help="FAISS 인덱스 디렉터리")
    ap.add_argument("--query", required=True, help="검색할 질문 문장")
    ap.add_argument("--top-k", type=int, default=5, help="반환할 상위 결과 개수")
    ap.add_argument("--corp-name", default=None, help="기업명 필터 (예: 삼성전자)")
    ap.add_argument("--doc-type", default=None, help="서식 필터 (예: 거래소공시, 정기공시, 주요사항보고서, 지분공시)")
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
    )

    print(f"
===== 검색 결과 (질문: '{args.query}') =====")
    if not results:
        print("조건에 맞는 검색 결과가 없습니다.")
        return

    for rank, item in enumerate(results, start=1):
        print(f"
[결과 {rank}] 유사도 점수: {item.get('score', 0.0):.4f}")
        print(f"- 기업명: {item.get('corp_name')} ({item.get('stock_code')}) | 서식: {item.get('source_doc_type')}")
        print(f"- 보고서: {item.get('report_nm')} (접수일: {item.get('rcept_dt')})")
        print(f"- 섹션  : {item.get('section') or item.get('section_path')}")
        print(f"- 청크ID: {item.get('chunk_id')}")
        print(f"--- [본문 내용] ---")
        text = item.get("text", "")
        print(text[:300] + ("..." if len(text) > 300 else ""))
        print("-" * 50)


if __name__ == "__main__":
    main()
