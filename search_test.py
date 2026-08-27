import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. 파일 경로 설정 (방금 완성한 파일들)
index_path = 'faiss_index/index.faiss'
metadata_path = 'faiss_index/metadata_final.pkl'

# 2. 모델, 인덱스, 메타데이터 로드
print("🤖 AI 모델 로드 중... (최초 실행 시 몇 초 걸릴 수 있습니다)")
model = SentenceTransformer('jhgan/ko-sroberta-multitask')

print("📂 검색 엔진(FAISS)과 데이터베이스 불러오는 중...")
index = faiss.read_index(index_path)

with open(metadata_path, 'rb') as f:
    metadata = pickle.load(f)

print(f"✅ 준비 완료! (총 {index.ntotal}개의 데이터가 대기 중입니다)\n")

# 3. 검색 엔진 함수
def search_engine(query, k=3):
    print(f"\n🔍 질문: '{query}'")
    print("=" * 50)
    
    # 질문을 AI 벡터 숫자로 변환
    query_vector = model.encode([query]).astype('float32')
    
    # FAISS 인덱스에서 가장 유사한 데이터 k개 찾기
    distances, indices = index.search(query_vector, k)
    
    # 결과 출력
    for i, idx in enumerate(indices[0]):
        score = distances[0][i]
        result_data = metadata[idx]
        text = result_data.get('text', '내용 없음')
        
        print(f"[{i+1}위] (유사도: {score:.4f})")
        print(f"💡 내용: {text[:300]}...") # 너무 길면 300자에서 자르기
        print("-" * 50)

# 4. 실전 테스트 무한 루프
if __name__ == "__main__":
    while True:
        user_query = input("\n💬 검색할 내용을 입력하세요 (종료하려면 q 입력): ")
        if user_query.lower() == 'q':
            print("검색을 종료합니다.")
            break
        
        search_engine(user_query, k=3)