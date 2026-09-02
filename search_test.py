import os
import requests
import faiss
import pickle
import numpy as np
import json
import uuid
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from datetime import datetime  # 💡 연도 계산을 위해 추가

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

print(f"📂 FAISS 인덱스 로드: {INDEX_PATH}")
index = faiss.read_index(index_path)

print(f"📂 메타데이터 로드: {METADATA_PATH}")
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

# ==========================================
# 💡 [신규 추가] 질문 재작성 (Query Rewriting) 함수
# ==========================================
def rewrite_query(original_query):
    """
    사용자 질문에 포함된 상대적 시간(올해, 작년 등)을 절대 연도로 변환합니다.
    """
    current_year = datetime.now().year
    
    system_prompt = f"""
당신은 검색 쿼리 변환 어시스턴트입니다.
현재 기준 연도는 {current_year}년입니다.
사용자의 질문에 포함된 상대적인 시점 표현을 아래 기준에 따라 정확한 연도로 변환하여 오직 '변환된 질문' 한 문장만 출력하세요. 부연 설명은 절대 추가하지 마세요.

[변환 기준]
- 올해 / 이번 년도 -> {current_year}년
- 작년 / 전년도 -> {current_year - 1}년
- 재작년 -> {current_year - 2}년
- 내년 -> {current_year + 1}년
- 특정 시점 언급이 없으면 원본 질문 유지

[예시]
- 질문: kb 금융 올해 매출은 얼마야? -> {current_year}년 KB금융 매출액은 얼마인가요?
- 질문: 작년 영업이익 알려줘 -> {current_year - 1}년 영업이익 알려줘
"""

    headers = {
        'Authorization': f'Bearer {NAVER_API_KEY}',
        'X-NCP-CLOVASTUDIO-REQUEST-ID': str(uuid.uuid4()),
        'Content-Type': 'application/json; charset=utf-8'
    }

    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": original_query}
        ],
        "temperature": 0.0,
        "maxTokens": 60
    }
    
    try:
        res = requests.post(NAVER_ENDPOINT, headers=headers, json=payload, timeout=5)
        if res.status_code == 200:
            result = res.json()
            return result.get('result', {}).get('message', {}).get('content', original_query).strip()
    except Exception:
        pass
    return original_query

# 3. 비서 역할(LLM) 함수 만들기
def generate_answer(query, context_text):
    system_prompt = """
    당신은 기업 공시를 분석하는 전문 AI 어시스턴트입니다. 
    반드시 제공된 [참고 자료]에 있는 정보만 사용하여 답변하세요.
    
    [필수 작성 규칙]
    1. 불필요한 서두와 결미(예: '안녕하세요', '죄송합니다만', '확인할 수 있는 내용은 다음과 같습니다')는 절대 출력하지 마세요.
    2. 질문에 대한 핵심 수치와 팩트를 1~2문장의 깔끔한 완성형 문장으로 바로 제시하세요.
    3. 숫자는 읽기 쉽게 단위(조, 억원 등)로 환산하여 표현해 주세요. (예: 333,605,938 백만원 -> 333조 6,059억 원)
    4. 답변 바로 다음 줄에 아래 포맷으로 간결하게 근거를 명시하세요.

    [중요 규칙: 무관한 정보 처리 (Anti-Hallucination)]
    사용자가 묻는 핵심 주제(예: '투자 내용', '매출액' 등)와 일치하는 내용이 [참고 자료]에 없다면, 절대 다른 내용을 억지로 템플릿에 끼워 맞추지 마세요. 
    이 경우 오직 아래 문장만 출력하세요.
    "검색된 자료에서 요청하신 '{질문 주제}'에 대한 구체적인 정보를 찾을 수 없습니다. 검색 키워드를 변경해 주세요."

    [답변 템플릿]
    {기업명} {연도/보고서명} 기준 주요 {질문 주제}:
    1. {핵심 항목1} — {수치 및 간결한 증감 요약}
    2. {핵심 항목2} — {수치 및 간결한 증감 요약}
    근거: 접수번호 {접수번호}, {목차명} / {목차명2}
    
    [작성 예시]
    KB금융 2025년 사업보고서 기준 주요 실적:
    1. 당기순이익 — 5조 8,332억 원 (전년 대비 7,550억 원 증가)
    2. 주요 계열사 실적 — 국민은행 3조 852억 원, KB손해보험 7,780억 원, KB증권 6,740억 원 등
    근거: 접수번호 20260619000667, II. 사업의 내용 - 마. 그룹 영업실적
    
    [중요 규칙: 정정공시 처리]
    1. 검색된 [참고 자료]에 과거 문서(정정 전)와 최신 문서(정정 후) 내용이 모두 포함되어 있다면, 최신 버전의 수치와 증감률을 우선적으로 반영하세요.
    2. 최신 문서(is_latest가 true인 문서)의 내용을 최종 사실로 간주하세요.
    """
    safe_context = context_text[:3500] if len(context_text) > 3500 else context_text
    user_prompt = f"질문: {query}\n\n[참고 자료]\n{context_text}"
    
    headers = {
        'Authorization': f'Bearer {NAVER_API_KEY}',
        'Content-Type': 'application/json; charset=utf-8'
    }
    
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.1,  # 수치 계산의 정확도를 위해 낮게 유지
        "maxTokens": 800     # 답변이 길어질 수 있으므로 토큰 여유를 조금 더 줍니다.
    }
    
    response = requests.post(NAVER_ENDPOINT, headers=headers, json=payload)
    
    if response.status_code == 200:
        result_data = response.json()
        return result_data['result']['message']['content']
    else:
        return f"🚨 API 호출 에러 발생 (상태 코드: {response.status_code})\n{response.text}"
    
# 4. 검색 + 답변 생성 통합 함수 (메타데이터 및 족보 결합 완결판)
def rag_search(query, k=20):
    print(f"\n🔍 원본 질문: '{query}'")
    
    # 💡 1. 쿼리 재작성 실행
    search_query = rewrite_query(query)
    if query != search_query:
        print(f"🔄 시점 변환 완료: '{search_query}'")
        
    print(f"🔍 '{search_query}'에 대한 분석을 시작합니다...")
    
    # 💡 2. 재작성된 쿼리로 벡터 검색 수행
    query_vector = model.encode([search_query]).astype('float32')
    distances, indices = index.search(query_vector, k)
    
    retrieved_texts = []
    for idx in indices[0]:
        chunk = metadata[idx]
        corp_name = chunk.get('corp_name', '')
        
        # 💡 [필터링 추가] 질문에 특정 기업명이 언급되었는데, 검색된 청크의 기업명과 다르면 버림
        if corp_name and (corp_name[:2] not in search_query) and (any(kw in search_query for kw in ["삼성", "SK", "KB", "카카오", "현대"])):
            continue # 다른 기업 정보는 건너뜀
        
        text = chunk.get('text', '내용 없음')
        doc_id = chunk.get('doc_id', '')
        section = chunk.get('heading_path', '목차 없음')
        
        receipt_no = doc_id.split('_')[-1] if '_' in doc_id else doc_id
        
        lineage_info = doc_lineage.get(receipt_no, {})
        
        if lineage_info:
            lineage_corp_name = lineage_info.get('corp_name', '알 수 없는 기업')
            
            report_nm = "보고서"
            for history_item in lineage_info.get('history', []):
                if history_item.get('rcept_no') == receipt_no:
                    report_nm = history_item.get('report_nm', '보고서')
                    break
                    
            file_name = f"{lineage_corp_name} {report_nm}" 
        else:
            file_name = doc_id 
        
        formatted_text = f"--- 검색된 문서 ---\n[출처 정보: 파일명 '{file_name}', 접수번호 '{receipt_no}', 목차 '{section}']\n"
        
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
    # 💡 3. 최종 답변 생성 (재작성된 쿼리 전달)
    final_answer = generate_answer(search_query, combined_context)
    
    print("=" * 50)
    print(final_answer)
    print("=" * 50)

# 실행
if __name__ == "__main__":
    while True:
        user_query = input("\n💬 질문 (q 입력 시 종료): ")
        if user_query.lower() == 'q':
            break
        rag_search(user_query, k=20)
    
