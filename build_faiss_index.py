# -*- coding: utf-8 -*-
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


def load_lineage(lineage_path: Path) -> dict:
    if not lineage_path.exists():
        print(f'[WARN] 계보 파일({lineage_path})을 찾을 수 없습니다. 기본값(is_latest=True)으로 진행합니다.')
        return {}
    with open(lineage_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_index(embeddings_path: str | Path, lineage_path: str | Path, out_dir: str | Path):
    if faiss is None:
        print('[ERROR] faiss-cpu 라이브러리가 필요합니다. 먼저 pip install faiss-cpu 를 실행하세요.', file=sys.stderr)
        sys.exit(1)

    embeddings_path = Path(embeddings_path)
    lineage_path = Path(lineage_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not embeddings_path.exists():
        raise FileNotFoundError(f'임베딩 파일을 찾을 수 없습니다: {embeddings_path}')

    lineage_map = load_lineage(lineage_path)

    print(f'[1/4] 임베딩 파일 및 계보 메타데이터 로드 중: {embeddings_path}')
    vectors = []
    metadata = []
    n_corrupt = 0

    t0 = time.time()
    with open(embeddings_path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception as e:
                n_corrupt += 1
                if n_corrupt <= 5:
                    print(f'[WARN] 라인 {line_no} JSON 파싱 실패 (스킵): {e}', file=sys.stderr)
                continue

            emb = rec.pop('embedding', None)
            if emb is None:
                continue

            r_no = str(rec.get('rcept_no', ''))
            lin_info = lineage_map.get(r_no, {})

            rec['lineage_id'] = lin_info.get('lineage_id', f'lineage_{r_no}')
            rec['is_latest'] = lin_info.get('is_latest', True)
            rec['version_order'] = lin_info.get('version_order', 1)
            rec['total_versions'] = lin_info.get('total_versions', 1)
            rec['history'] = lin_info.get('history', [])

            vectors.append(emb)
            metadata.append(rec)

            if line_no % 5000 == 0:
                print(f'      {line_no}건 처리 완료...')

    n_total = len(vectors)
    if n_total == 0:
        raise ValueError('임베딩 데이터가 비어 있습니다.')

    dim = len(vectors[0])
    print(f'[2/4] NumPy 행렬 변환 중... (총 {n_total:,}개 청크, {dim}차원, 손상 라인: {n_corrupt}건)')
    v_matrix = np.array(vectors, dtype=np.float32)

    print('[3/4] 벡터 L2 정규화 (코사인 유사도 검색용)...')
    faiss.normalize_L2(v_matrix)

    print('[4/4] FAISS IndexFlatIP 인덱스 구축 및 파일 저장...')
    index = faiss.IndexFlatIP(dim)
    index.add(v_matrix)

    index_path = out_dir / 'index.faiss'
    faiss.write_index(index, str(index_path))

    meta_path = out_dir / 'metadata.pkl'
    with open(meta_path, 'wb') as f:
        pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

    elapsed = time.time() - t0
    n_latest = sum(1 for m in metadata if m.get('is_latest', True))
    n_old = n_total - n_latest

    print()
    print('===== FAISS 인덱스 구축 완료 (계보 메타데이터 주입) =====')
    print(f'총 청크 수           : {n_total:,} 개')
    print(f'- 최종 최신본 청크  : {n_latest:,} 개 (is_latest=True)')
    print(f'- 과거 구버전 청크  : {n_old:,} 개 (정정 전 원본)')
    print(f'인덱스 파일         : {index_path} ({index_path.stat().st_size / (1024*1024):.2f} MB)')
    print(f'메타데이터 파일     : {meta_path} ({meta_path.stat().st_size / (1024*1024):.2f} MB)')
    print(f'소요 시간           : {elapsed:.2f} 초')


def main():
    ap = argparse.ArgumentParser(description='embeddings.jsonl + doc_lineage.json -> FAISS 인덱스 빌드')
    ap.add_argument('--embeddings', required=True, help='embed_chunks.py 결과물 (embeddings.jsonl)')
    ap.add_argument('--lineage', default='doc_lineage.json', help='build_doc_lineage.py 결과물 (doc_lineage.json)')
    ap.add_argument('--out-dir', default='faiss_index', help='FAISS 인덱스 및 메타데이터 저장 디렉터리')
    args = ap.parse_args()

    build_index(args.embeddings, args.lineage, args.out_dir)


if __name__ == '__main__':
    main()
