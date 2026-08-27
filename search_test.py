import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ==========================================
# 🔑 OpenAI API 키 입력 (본인의 키로 변경하세요!)
# ==========================================
client = OpenAI(api_key="")

# 1. 파일 경로 설정
index_path = 'faiss_index/index.faiss'
metadata_path = 'faiss_index/metadata_final.pkl'

# 2. 모델, 인덱스, 메타데이터 로드
print("🤖 AI 모델 로드 중...")
model = SentenceTransformer('jhgan/ko-sroberta-multitask')
index = faiss.read_index(index_path)

with open(metadata_path, 'rb') as f:
    metadata = pickle.load(f)

print(f"✅ 준비 완료! (총 {index.ntotal}개의 데이터)")

# 3. 비서 역할(LLM) 함수 만들기
def generate_answer(query, context_text):
    # AI에게 역할과 규칙을 부여하는 프롬프트 엔지니어링!
    system_prompt = """
    당신은 기업 공시를 분석하는 친절한 AI 어시스턴트입니다. 
    반드시 제공된 [참고 자료]에 있는 정보만 사용하여 답변하세요.
    사용자가 원하는 형식(예: '당기순이익은 ~입니다. 작년 대비 ~증감했습니다.')에 최대한 맞춰서 답변을 가공해 주세요.
    만약 참고 자료에 작년도 데이터 등 비교할 수 있는 정보가 없다면, "제공된 자료에서는 작년도 데이터를 확인할 수 없어 증감 여부는 알 수 없습니다."라고 솔직하게 말하세요.
    """
    
    user_prompt = f"질문: {query}\n\n[참고 자료]\n{context_text}"
    
    response = client.chat.completions.create(
        model="gpt-4o-mini", # 가볍고 빠르고 저렴한 모델
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0 # 창의성 0 (상상해서 거짓말하지 않도록)
    )
    return response.choices[0].message.content

# 4. 검색 + 답변 생성 통합 함수
def rag_search(query, k=3):
    print(f"\n🔍 '{query}'에 대한 분석을 시작합니다...")
    
    # 1단계: FAISS로 문서를 검색 (사서 역할)
    query_vector = model.encode([query]).astype('float32')
    distances, indices = index.search(query_vector, k)
    
    # 검색된 텍스트들을 하나의 긴 문자열로 뭉치기
    retrieved_texts = []
    for idx in indices[0]:
        retrieved_texts.append(metadata[idx].get('text', ''))
    
    combined_context = "\n\n---\n\n".join(retrieved_texts)
    
    # 2단계: OpenAI(비서)에게 요약 요청!
    print("✨ AI 비서가 문서를 읽고 답변을 작성 중입니다...\n")
    final_answer = generate_answer(query, combined_context)
    
    print("=" * 50)
    print(final_answer)
    print("=" * 50)

# 실행
if __name__ == "__main__":
    while True:
        user_query = input("\n💬 질문 (q 입력 시 종료): ")
        if user_query.lower() == 'q':
            break
        rag_search(user_query, k=3)
