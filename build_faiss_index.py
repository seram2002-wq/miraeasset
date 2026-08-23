# -*- coding: utf-8 -*-
"""
build_faiss_index.py
=====================================================================
[코드 설명]
  - embed_chunks.py 에서 Clova API로 임베딩하여 만든 벡터 파일(embeddings.jsonl)을 읽어
    초고속 벡터 검색기인 FAISS 인덱스(index.faiss)와 메타데이터 장부(metadata.pkl)를 구축합니다.
  - 벡터는 코사인 유사도(Cosine Similarity) 측정을 위해 L2 정규화 후 IndexFlatIP 인덱스에 저장됩니다.

[필요 라이브러리 설치]
  pip install faiss-cpu numpy

[실행 방법]
  python build_faiss_index.py --embeddings embeddings.jsonl --out-dir faiss_index

[주요 인자(Arguments)]
  --embeddings : embed_chunks.py가 생성한 embeddings.jsonl 파일 경로 (필수)
  --out-dir    : 인덱스와 메타데이터가 저장될 폴더 경로 (기본값: faiss_index)

[주의사항]
  1. FAISS는 숫자 벡터만 기억하므로, 질문 검색 시 원본 텍스트와 회사명 등을 복원하기 위해
     metadata.pkl 파일이 index.faiss 와 1:1로 항상 같은 폴더에 함께 보관되어야 합니다.
  2. 대용량 임베딩 파일(수만 건)도 수 초 내에 고속으로 인덱싱됩니다.
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
import numpy as np

try:
    import faiss
except ImportError:
    faiss = None


def build_index(embeddings_path: str | Path, out_dir: str | Path):
    if faiss is None:
        print("[ERROR] faiss 라이브러리가 필요합니다. 먼저 pip install faiss-cpu 를 실행하세요.", file=sys.stderr)
        sys.exit(1)

    embeddings_path = Path(embeddings_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not embeddings_path.exists():
        raise FileNotFoundError(f"임베딩 파일을 찾을 수 없습니다: {embeddings_path}")

    print(f"[1/4] 임베딩 파일 읽는 중...: {embeddings_path}")
    vectors = []
    metadata = []

    t0 = time.time()
    with open(embeddings_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            emb = rec.pop("embedding", None)
            if emb is None:
                continue
            vectors.append(emb)
            metadata.append(rec)

            if line_no % 5000 == 0:
                print(f"      {line_no}건 로드 완료...")

    n_total = len(vectors)
    if n_total == 0:
        raise ValueError("임베딩 데이터가 비어 있습니다.")

    dim = len(vectors[0])
    print(f"[2/4] NumPy 행렬 변환 중... (총 {n_total:,}개 청크, {dim}차원)")
    v_matrix = np.array(vectors, dtype=np.float32)

    print("[3/4] 벡터 L2 정규화 (코사인 유사도 검색용)...")
    faiss.normalize_L2(v_matrix)

    print("[4/4] FAISS IndexFlatIP 인덱스 구축 및 파일 저장...")
    index = faiss.IndexFlatIP(dim)
    index.add(v_matrix)

    index_path = out_dir / "index.faiss"
    faiss.write_index(index, str(index_path))

    meta_path = out_dir / "metadata.pkl"
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

    elapsed = time.time() - t0
    print("
===== FAISS 인덱스 구축 완료 =====")
    print(f"총 청크 수      : {n_total:,} 개")
    print(f"임베딩 차원    : {dim} 차원")
    print(f"인덱스 파일    : {index_path} ({index_path.stat().st_size / (1024*1024):.2f} MB)")
    print(f"메타데이터 파일: {meta_path} ({meta_path.stat().st_size / (1024*1024):.2f} MB)")
    print(f"소요 시간      : {elapsed:.2f} 초")


def main():
    ap = argparse.ArgumentParser(description="embeddings.jsonl -> FAISS 인덱스 빌드")
    ap.add_argument("--embeddings", required=True, help="embed_chunks.py 결과물 (embeddings.jsonl)")
    ap.add_argument("--out-dir", default="faiss_index", help="FAISS 인덱스 및 메타데이터 저장 디렉터리")
    args = ap.parse_args()

    build_index(args.embeddings, args.out_dir)


if __name__ == "__main__":
    main()
