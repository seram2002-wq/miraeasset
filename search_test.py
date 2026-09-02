import os
import re
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

print(f"📂 FAISS 인덱스 로드: {index_path}")
index = faiss.read_index(index_path)

print(f"📂 메타데이터 로드: {metadata_path}")
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
# 🔧 [수정 1] 동적 회사명 매칭 준비
# 기존: corp_name[:2] not in search_query 방식 (앞 2글자만 비교) + 5개 기업만 하드코딩
# 문제: "삼성전자"질문에 "삼성중공업/삼성생명"등도 다같이 통과됨
#       "NAVER"처럼 하드코딩 리스트에 없는 회사는 필터 자체가 작동 안 함
# 해결: metadata에 실제 존재하는 모든 corp_name을 기준으로 정확 매칭
# ==========================================
ALL_CORP_NAMES = sorted(
    set(chunk.get('corp_name', '') for chunk in metadata if chunk.get('corp_name')),
    key=len, reverse=True  # 긴 이름부터 비교해야 "삼성전자"가 "삼성"보다 먼저 매칭됨
)

# 영문/약칭 등 별칭 매핑 (필요할 때마다 계속 추가하면 됩니다)
CORP_ALIASES = {
    "네이버": "NAVER",
    "삼성전자": "삼성전자",
    # 예: "SK하닉": "SK하이닉스", "포스코": "POSCO홀딩스" 등
}

def extract_target_corp(query, corp_names, aliases):
    """질문 문장에서 실제 metadata에 존재하는 기업명을 찾아 반환합니다."""
    for c in corp_names:
        if c and c in query:
            return c
    for alias, real in aliases.items():
        if alias in query:
            return real
    return None

# ==========================================
# 🔧 [수정 11] 신규 추가 - 문서별 목차(섹션) 사전 인덱싱
# 문제: "투자 계획"처럼 한 단어가 보고서 안에서 여러 의미(설비투자 vs 지분투자/M&A)로
#       쓰이면, 임베딩 검색만으로는 엉뚱한 섹션(예: III.재무제표 주석의 지분 매각/인수 내용)을
#       "II-3.원재료및생산설비/II-6.연구개발활동" 대신 끌고 올 수 있음
# 해결: 문서(doc_id)별 실제 존재하는 heading_path 목록을 미리 모아두고,
#       질문에 맞는 섹션을 LLM으로 먼저 고르게 한 뒤 그 섹션만 사용하도록 함 (아래 수정 12)
# ==========================================
# ==========================================
# 🔧 [수정 15] 신규 추가 - "접수연도" 대신 "보고 대상 기간(회계연도)" 기준으로 연도 판단
# 문제: 사업보고서는 회계연도 종료 후 다음 해 3월에 제출됨
#       예) "사업보고서 (2025.12)" = 2025년 실적이지만 접수는 2026년 3월
#       기존 방식(receipt_no[:4] == 접수연도)으로 연도를 비교하면
#       "2025년 매출"을 물었을 때 정작 2025년 실적이 담긴 사업보고서가
#       접수연도가 2026이라는 이유로 필터에서 통째로 걸러짐
# 해결: doc_lineage.json의 report_nm에 있는 "(YYYY.MM)" 표기(=실제 보고 대상 기간)를
#      파싱해서 연도 판단 기준으로 사용. report_nm이 없으면 접수연도로 대체(fallback)
# ==========================================
def get_period_year(receipt_no):
    lineage_info = doc_lineage.get(receipt_no, {})
    report_nm = None
    for h in lineage_info.get('history', []):
        if h.get('rcept_no') == receipt_no:
            report_nm = h.get('report_nm', '')
            break
    if report_nm:
        m = re.search(r'\((\d{4})\.\d{2}\)', report_nm)
        if m:
            return m.group(1)
    # report_nm에 기간 표기가 없으면 접수연도로 대체
    return receipt_no[:4] if receipt_no[:4].isdigit() else None


DOC_SECTIONS = {}  # doc_id -> {heading_path, heading_path, ...}
DOC_INFO = {}       # doc_id -> {'corp_name':.., 'year':..}
for chunk in metadata:
    _doc_id = chunk.get('doc_id')
    if not _doc_id:
        continue
    _heading = chunk.get('heading_path')
    if _heading:
        DOC_SECTIONS.setdefault(_doc_id, set()).add(_heading)
    if _doc_id not in DOC_INFO:
        _receipt_no = _doc_id.split('_')[-1] if '_' in _doc_id else _doc_id
        _year = get_period_year(_receipt_no)  # 🔧 [수정 15] 접수연도 대신 보고기간 연도 사용
        DOC_INFO[_doc_id] = {'corp_name': chunk.get('corp_name', ''), 'year': _year}


def find_target_doc_ids(target_corp, target_year, target_report_type):
    """회사/연도/보고서종류 조건에 맞는 문서(doc_id)들을 찾음"""
    matched = []
    for doc_id, info in DOC_INFO.items():
        if target_corp and info['corp_name'] != target_corp:
            continue
        if target_year and info['year'] != target_year:
            continue
        if target_report_type:
            headings = DOC_SECTIONS.get(doc_id, set())
            if target_report_type not in doc_id and not any(target_report_type in h for h in headings):
                continue
        matched.append(doc_id)
    return matched


def select_relevant_sections(query, available_sections):
    """문서의 목차 리스트 중, 질문에 답하기 위해 봐야 할 섹션을 LLM으로 선택합니다."""
    if not available_sections:
        return []

    sections_list_str = "\n".join(f"- {s}" for s in sorted(available_sections))
    system_prompt = f"""당신은 기업 공시 보고서의 목차 중 질문과 가장 관련 있는 섹션을 고르는 어시스턴트입니다.
아래는 이 보고서의 전체 목차 목록입니다:
{sections_list_str}

사용자 질문에 답하기 위해 반드시 확인해야 하는 섹션을 목록에서 최대 3개까지 고르세요.
목록에 있는 텍스트를 정확히 그대로 복사해서, 콤마(,)로 구분해 출력하세요. 그 외 부연 설명은 절대 추가하지 마세요."""

    headers = {
        'Authorization': f'Bearer {NAVER_API_KEY}',
        'X-NCP-CLOVASTUDIO-REQUEST-ID': str(uuid.uuid4()),
        'Content-Type': 'application/json; charset=utf-8'
    }
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        "temperature": 0.0,
        "maxTokens": 200
    }
    try:
        res = requests.post(NAVER_ENDPOINT, headers=headers, json=payload, timeout=5)
        if res.status_code != 200:
            print(f"⚠️ [섹션 선택 API 오류] 상태 코드: {res.status_code}")  # 🔧 [수정 17]
            return []
        result = res.json()
        content = result.get('result', {}).get('message', {}).get('content', '').strip()
        picked = [s.strip() for s in content.split(',') if s.strip()]
        # 🔧 [수정 16] 완전 일치 우선, 실패 시 부분 일치(포함 관계)까지 허용
        #     이유: LLM이 목차를 그대로 베끼다가 공백/번호 표기가 살짝 달라지면
        #          완전 일치 검증에서 전부 탈락해 target_sections가 비어버리는 문제 방지
        valid = [s for s in picked if s in available_sections]
        if not valid:
            for p in picked:
                for s in available_sections:
                    if p and (p in s or s in p):
                        valid.append(s)
        if not valid:
            print(f"⚠️ [섹션 선택 실패] LLM 응답: '{content}' / 목차 후보 {len(available_sections)}개 중 매칭 없음")  # 🔧 [수정 17]
        return list(dict.fromkeys(valid))  # 순서 유지하며 중복 제거
    except Exception as e:
        print(f"⚠️ [섹션 선택 예외] {e}")  # 🔧 [수정 17]
        return []

# 🔧 [수정 5] 신규 추가 - 질문에 포함된 연도(시점) 추출
# rewrite_query가 이미 "2026년" 같은 절대 연도로 바꿔주므로, 여기서는 4자리 연도만 뽑으면 됨
def extract_target_year(query):
    match = re.search(r'(20\d{2})년', query)
    return match.group(1) if match else None

# 🔧 [수정 8] 신규 추가 - 질문에 포함된 보고서 종류(분기/반기/사업보고서) 추출
# 예: "2026년 1분기 분기보고서" -> "분기보고서"로 매칭되도록
REPORT_TYPE_KEYWORDS = {
    "분기보고서": ["분기보고서", "1분기", "2분기", "3분기", "4분기", "분기 실적"],
    "반기보고서": ["반기보고서", "반기 실적"],
    "사업보고서": ["사업보고서", "연간 실적", "연간보고서"],
}

def extract_target_report_type(query):
    for report_type, keywords in REPORT_TYPE_KEYWORDS.items():
        if any(kw in query for kw in keywords):
            return report_type
    return None

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
    🔧 [수정 4] 기존에는 질문 주제와 '정확히' 일치하지 않으면 무조건 거절 문구만 출력했음
              -> 랜덤 질문에서 "관련은 있지만 완전 일치는 아닌" 자료도 전부 거절되는 원인이었음
    변경된 규칙:
    - [참고 자료]에 질문과 관련성 있는 내용이 조금이라도 있다면, 그 범위 내에서 최대한 답변하세요.
      그리고 답변 마지막 줄에 "다만 요청하신 정확한 항목({세부 항목})은 자료에서 확인되지 않았습니다."처럼
      부족한 부분만 짧게 덧붙이세요.
    - [참고 자료] 전체가 질문 주제와 완전히 무관할 때만 아래 거절 문장을 출력하세요.
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
    
# ==========================================
# 🔧 [수정 13] 신규 추가 - 청크 포맷팅 로직을 함수로 분리
# 기존엔 이 로직이 rag_search 안 for문 안에 박혀 있어서, 벡터검색 결과에만 적용 가능했음
# 분리 이유: 아래에서 "섹션 직접 조회" 경로도 동일한 포맷을 써야 하기 때문
# ==========================================
def format_chunk(chunk):
    text = chunk.get('text', '내용 없음')
    doc_id = chunk.get('doc_id', '')
    section = chunk.get('heading_path', '목차 없음')
    corp_name = chunk.get('corp_name', '')

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
    return formatted_text, corp_name, receipt_no, section, file_name


# 4. 검색 + 답변 생성 통합 함수 (메타데이터 및 족보 결합 완결판)
def rag_search(query, k=20):
    print(f"\n🔍 원본 질문: '{query}'")
    
    # 💡 1. 쿼리 재작성 실행
    search_query = rewrite_query(query)
    if query != search_query:
        print(f"🔄 시점 변환 완료: '{search_query}'")
        
    print(f"🔍 '{search_query}'에 대한 분석을 시작합니다...")
    
    # 🔧 [수정 9] 벡터 검색 전에 대상 기업/연도/보고서종류를 먼저 추출
    #     기존 문제: k=20으로 "전체 인덱스"에서 상위 20개만 뽑다 보니,
    #               삼성전자만 13,257개 청크가 있는 상황에서 원하는 특정 보고서(739개)의
    #               특정 내용 청크가 top-20 안에 아예 안 들어오는 경우가 많았음
    #               (필터링은 top-20을 뽑은 "이후"에 적용되므로,애초에 후보군에 없으면 필터도 소용없음)
    #     해결: 회사/연도/보고서종류처럼 특정 조건이 잡히면, k를 훨씬 크게(예: 300) 잡아서
    #          후보군 자체를 넓힌 뒤 필터링. 조건이 없는 일반 질문은 기존 k(기본 20) 그대로 사용
    target_corp = extract_target_corp(search_query, ALL_CORP_NAMES, CORP_ALIASES)
    target_year = extract_target_year(search_query)
    target_report_type = extract_target_report_type(search_query)

    search_k = k
    if target_corp:  # 특정 조건이 있으면 후보군을 넓게 확보
        search_k = max(k, 300)

    if target_corp:
        print(f"🏢 인식된 대상 기업: '{target_corp}'")
    if target_year:
        print(f"📅 인식된 대상 연도: '{target_year}'")
    if target_report_type:
        print(f"📑 인식된 대상 보고서 종류: '{target_report_type}'")
    if search_k != k:
        print(f"🔎 조건이 특정되어 검색 범위를 k={search_k}로 확대합니다.")

    # 🔧 [수정 12] 신규 추가 - "섹션 선택" 단계
    #     문제: "투자 계획" 같은 질문이 보고서 내 여러 섹션(설비투자 vs 지분투자/M&A 등)에
    #          걸쳐 나타날 때, 임베딩 유사도만으론 의도한 섹션(II-3,II-6)이 아니라
    #          엉뚱한 섹션(III.재무제표주석, XII.상세표 등)이 뽑힐 수 있음
    #     해결: 대상 문서가 특정되면, 그 문서의 전체 목차를 LLM에 보여주고
    #          질문과 관련된 섹션을 먼저 고르게 함. 이후 그 섹션의 청크만 사용
    target_sections = []
    target_doc_ids = []
    if target_corp:
        target_doc_ids = find_target_doc_ids(target_corp, target_year, target_report_type)
        if not target_doc_ids:
            print(f"⚠️ [문서 조회 실패] '{target_corp} {target_year or ''} {target_report_type or ''}' 조건에 맞는 문서가 없습니다.")
        if target_doc_ids:
            available_sections = set()
            for d in target_doc_ids:
                available_sections |= DOC_SECTIONS.get(d, set())
            target_sections = select_relevant_sections(search_query, available_sections)
            if target_sections:
                print(f"📂 선택된 섹션: {target_sections}")

    # 🔧 [수정 14] 신규 추가 - "섹션 직접 조회" 경로
    #     문제: 섹션까지 특정됐어도, 그 섹션의 청크가 벡터검색 top-300 순위 안에
    #          다 들어온다는 보장은 없음 (일부만 걸릴 수도 있음 -> 답변 누락 위험)
    #     해결: target_doc_ids + target_sections가 모두 확정되면, 벡터검색 없이
    #          메타데이터에서 해당 문서+섹션 청크를 통째로 직접 긁어옴 (정확도/재현율 100%)
    #     실패(섹션 선택이 안 됐거나 매칭 결과가 없음)하면 기존 벡터검색 경로로 자동 폴백
    retrieved_texts = []
    filtered_out_texts = []
    MAX_CONTEXT_CHUNKS = 20

    if target_sections and target_doc_ids:
        direct_chunks = [
            c for c in metadata
            if c.get('doc_id') in target_doc_ids and c.get('heading_path') in target_sections
        ]
        print(f"📌 섹션 직접 조회로 {len(direct_chunks)}개 청크를 찾았습니다.")
        for c in direct_chunks[:MAX_CONTEXT_CHUNKS]:
            formatted_text, _, _, _, _ = format_chunk(c)
            retrieved_texts.append(formatted_text)

    if retrieved_texts:
        combined_context = "\n".join(retrieved_texts)
        print("✨ AI 비서가 문서를 비교 분석하여 답변을 작성 중입니다...\n")
        final_answer = generate_answer(search_query, combined_context)
        print("=" * 50)
        print(final_answer)
        print("=" * 50)
        return  # 🔧 직접 조회로 답을 만들었으면 아래 벡터검색 경로는 건너뜀

    # ---- 여기부터는 섹션 직접 조회가 불가능했던 경우(target_sections가 없거나 결과 0건)의
    #      기존 벡터검색 기반 경로 ----

    # 💡 2. 재작성된 쿼리로 벡터 검색 수행
    query_vector = model.encode([search_query]).astype('float32')
    distances, indices = index.search(query_vector, search_k)

    # 🔧 [수정 10] search_k를 300까지 늘렸으므로, 필터를 통과한 청크가
    #     너무 많아지면 LLM 컨텍스트가 과도하게 커질 수 있음. 상한선을 둬서 방지.
    #     (indices는 거리순 정렬이라 앞에서부터 채우면 자연스럽게 가장 유사한 것부터 담김)
    for idx in indices[0]:
        if len(retrieved_texts) >= MAX_CONTEXT_CHUNKS:
            break
        chunk = metadata[idx]
        formatted_text, corp_name, receipt_no, section, file_name = format_chunk(chunk)

        # 🔧 [수정 7, 15] receipt_no 앞 4자리(접수연도) 대신 회계기간 연도(get_period_year)로 비교
        chunk_year = get_period_year(receipt_no)
        corp_mismatch = target_corp and corp_name and corp_name != target_corp
        year_mismatch = target_year and chunk_year and chunk_year != target_year

        # 🔧 [수정 8] heading_path/file_name에 보고서 종류가 포함되는지 체크
        report_type_mismatch = (
            target_report_type
            and target_report_type not in section
            and target_report_type not in file_name
        )

        # 🔧 [수정 12] 선택된 섹션과 다르면 걸러냄 (여기 도달했다는 건 target_sections가
        #     비어있거나 직접조회가 0건이었다는 뜻이라 사실상 무력화됨)
        section_mismatch = target_sections and section not in target_sections

        if corp_mismatch or year_mismatch or report_type_mismatch or section_mismatch:
            filtered_out_texts.append(formatted_text)
            continue

        retrieved_texts.append(formatted_text)
    
    # 🔧 [수정 3] Fallback 로직
    #     기존 문제: 필터 통과 결과가 0개면, 시점이 완전히 다른 문서(예: 2023년 보고서)를
    #               아무 표시 없이 그대로 답변 재료로 써버려서 LLM이 헷갈릴 여지가 있었음
    #     변경: fallback으로 원본 결과를 쓰더라도, "요청하신 정확한 시점/기업의 자료가 아님"을
    #          컨텍스트 맨 앞에 명시적으로 박아넣어 LLM이 반드시 그 사실을 답변에 반영하게 함
    if not retrieved_texts and filtered_out_texts:
        print("⚠️ 대상 기업/연도로 필터링된 결과가 없어, 필터 없이 원본 검색 결과를 사용합니다.")
        warning_note = (
            f"⚠️⚠️ [시스템 경고] 요청하신 '{target_corp or ''} {target_year or ''} {target_report_type or ''}' 기준 자료가 "
            f"검색 인덱스에 존재하지 않습니다. 아래는 참고용으로 가장 유사한 다른 시점/자료이며, "
            f"질문에 대한 정확한 답이 아닐 수 있습니다. 반드시 이 사실을 답변에 명시하세요.\n\n"
        )
        retrieved_texts = [warning_note] + filtered_out_texts[:MAX_CONTEXT_CHUNKS]

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
