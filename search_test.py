import os
import requests
import faiss
import pickle
import numpy as np
import json  # json 파일 읽기를 위해 추가
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# ==========================================
# 🔑 .env 파일에서 환경 변수 로드
# ==========================================
load_dotenv()

NAVER_API_KEY = os.getenv("NAVER_API_KEY")
NAVER_APIGW_KEY = os.getenv("NAVER_APIGW_KEY")
NAVER_ENDPOINT = os.getenv("NAVER_ENDPOINT")

# 1. 파일 경로 설정
index_path = 'faiss_index/index.faiss'
metadata_path = 'faiss_index/metadata_final.pkl'
lineage_path = 'doc_lineage.json'  # 정정공시 이력(족보) 파일 경로

# 2. 모델, 인덱스, 메타데이터, 족보 데이터 로드
print("🤖 AI 모델 및 데이터 로드 중...")
model = SentenceTransformer('jhgan/ko-sroberta-multitask')
index = faiss.read_index(index_path)

with open(metadata_path, 'rb') as f:
    metadata = pickle.load(f)

# doc_lineage.json 로드
try:
    with open(lineage_path, 'r', encoding='utf-8') as f:
        doc_lineage = json.load(f)
    print("✅ 정정공시 이력(doc_lineage.json) 로드 완료!")
except FileNotFoundError:
    print("⚠️ doc_lineage.json 파일을 찾을 수 없습니다. 정정공시 비교 기능이 제한될 수 있습니다.")
    doc_lineage = {}

print(f"✅ 준비 완료! (총 {index.ntotal}개의 벡터 데이터)")

# 3. 비서 역할(LLM) 함수 만들기
def generate_answer(query, context_text):
    system_prompt = """
    당신은 기업 공시를 분석하는 전문적이고 친절한 AI 어시스턴트입니다. 
    반드시 제공된 [참고 자료]에 있는 정보만 사용하여 답변하세요.
    
    [중요 규칙: 정정공시 처리]
    1. 검색된 [참고 자료]에 과거 문서(정정 전)와 최신 문서(정정 후) 내용이 모두 포함되어 있다면, 두 내용을 비교하여 답변하세요.
       예시: "기존 분기보고서에서는 당기순이익이 100억 원이었으나, 이후 기재정정 공시를 통해 120억 원으로 정정되었습니다."
    2. 과거 문서에만 있는 내용이고 최신 문서에서 해당 내용이 삭제되었거나 다르게 적혀있다면, 최신 문서(is_latest가 true인 문서)의 내용을 최종 사실로 간주하세요.
    
    [중요 규칙: 출처 표기]
    답변을 모두 작성한 후, 맨 마지막 줄에 반드시 아래 형식을 지켜서 답변에 활용된 문서의 근거(출처)를 남겨주세요. 
    (여러 문서를 참고했다면 모두 적어주세요.)
    형식: (근거: "파일명", "접수번호 000000", "목차")
    """
    
    user_prompt = f"질문: {query}\n\n[참고 자료]\n{context_text}"
    
    # 💡 네이버 최신 인증 방식(Bearer)으로 헤더 수정
    headers = {
        'Authorization': f'Bearer {NAVER_API_KEY}',
        'Content-Type': 'application/json; charset=utf-8'
    }
    
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.1, 
        "maxTokens": 600
    }
    
    response = requests.post(NAVER_ENDPOINT, headers=headers, json=payload)
    
    if response.status_code == 200:
        result_data = response.json()
        return result_data['result']['message']['content']
    else:
        return f"🚨 API 호출 에러 발생 (상태 코드: {response.status_code})\n{response.text}"

    
# 4. 검색 + 답변 생성 통합 함수 (정정공시 이력 결합)
# 4. 검색 + 답변 생성 통합 함수 (메타데이터 및 족보 결합 완결판)
def rag_search(query, k=5):
    print(f"\n🔍 '{query}'에 대한 분석을 시작합니다...")
    
    query_vector = model.encode([query]).astype('float32')
    distances, indices = index.search(query_vector, k)
    
    retrieved_texts = []
    for idx in indices[0]:
        chunk = metadata[idx]
        
        # 1. 메타데이터에서 기본 정보 추출
        text = chunk.get('text', '내용 없음')
        doc_id = chunk.get('doc_id', '')
        section = chunk.get('heading_path', '목차 없음')
        
        # 2. doc_id에서 접수번호(숫자 부분)만 추출 (예: 'periodic_12345' -> '12345')
        receipt_no = doc_id.split('_')[-1] if '_' in doc_id else doc_id
        
        # 3. doc_lineage.json에서 족보 및 회사/보고서명 정보 가져오기
        lineage_info = doc_lineage.get(receipt_no, {})
        
        if lineage_info:
            corp_name = lineage_info.get('corp_name', '알 수 없는 기업')
            
            # history 목록을 뒤져서 현재 접수번호에 해당하는 보고서명 찾기
            report_nm = "보고서"
            for history_item in lineage_info.get('history', []):
                if history_item.get('rcept_no') == receipt_no:
                    report_nm = history_item.get('report_nm', '보고서')
                    break
                    
            file_name = f"{corp_name} {report_nm}" # 예: "삼성전자 반기보고서 (2023.06)"
        else:
            file_name = doc_id # 족보 정보가 없으면 doc_id를 임시 파일명으로 사용
        
        # 출처 텍스트 기본 포맷
        formatted_text = f"--- 검색된 문서 ---\n[출처 정보: 파일명 '{file_name}', 접수번호 '{receipt_no}', 목차 '{section}']\n"
        
        # 정정공시 여부 안내
        if lineage_info:
            is_latest = lineage_info.get("is_latest", False)
            if not is_latest:
                formatted_text += "⚠️ [주의: 이 문서는 과거 버전이며, 이후 접수된 기재정정 문서가 존재합니다.]\n"
            else:
                formatted_text += "✅ [안내: 이 문서는 기재정정이 반영되었거나 정정된 적 없는 최신 최종 버전입니다.]\n"
        else:
             formatted_text += "ℹ️ [안내: 이 문서에 대한 정정 이력 정보를 찾을 수 없습니다.]\n"
                
        formatted_text += f"문서 내용:\n{text}\n"
        retrieved_texts.append(formatted_text)
    
    combined_context = "\n".join(retrieved_texts)
    
    print("✨ AI 비서가 문서를 비교 분석하여 답변을 작성 중입니다...\n")
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
        rag_search(user_query, k=5)
