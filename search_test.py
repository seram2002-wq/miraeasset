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


ALL_CORP_NAMES = sorted(
    set(chunk.get('corp_name', '') for chunk in metadata if chunk.get('corp_name')),
    key=len, reverse=True  # 긴 이름부터 비교해야 "삼성전자"가 "삼성"보다 먼저 매칭됨
)

STOCK_CODE_MAP = {
    "000270": "기아","000660": "SK하이닉스","000720": "현대건설", "000810": "삼성화재해상보험",
    "001430": "세아베스틸지주","004020": "현대제철",    "005380": "현대자동차", "005490": "POSCO홀딩스",
    "005930": "삼성전자",    "006400": "삼성SDI",    "006800": "미래에셋증권",    "009150": "삼성전기",
    "009830": "한화솔루션",    "010060": "OCI홀딩스",    "010120": "엘에스일렉트릭",    "010130": "고려아연", "010140": "삼성중공업",
    "011070": "LG이노텍",    "011200": "HMM",    "012330": "현대모비스",    "012450": "한화에어로스페이스",
    "017670": "SK텔레콤",    "028050": "삼성E&A",    "030200": "케이티",    "032640": "LG유플러스",
    "032820": "우리기술",    "032830": "삼성생명",    "034020": "두산에너빌리티",    "035420": "NAVER",
    "035720": "카카오",    "036570": "NC",    "037620": "미래에셋증권",    "041510": "에스엠",
    "042660": "한화오션",   "042700": "한미반도체",    "047040": "대우건설",    "047810": "한국항공우주",
    "051900": "LG생활건강",    "052690": "한전기술",    "053000": "우리금융지주",    "055550": "신한지주",
    "064350": "현대로템",    "064400": "LG씨엔에스",    "068270": "셀트리온",    "079550": "LIG디펜스앤에어로스페이스",
    "086280": "현대글로비스",    "086790": "하나금융지주",    "090430": "아모레퍼시픽",    "097950": "CJ제일제당",
    "105560": "KB금융",    "122870": "와이지엔터테인먼트",    "128940": "한미약품",    "138040": "메리츠금융지주",
    "139480": "이마트",    "196170": "알테오젠",    "207940": "삼성바이오로직스",    "214450": "파마리서치",
    "247540": "에코프로비엠",    "259960": "크래프톤",    "267260": "HD현대일렉트릭",    "277810": "레인보우로보틱스",
    "298040": "효성중공업",    "307950": "현대오토에버",    "316140": "우리금융지주",    "329180": "HD현대중공업",
    "336260": "두산퓨얼셀",    "347850": "디앤디파마텍",    "352820": "하이브",    "373220": "LG에너지솔루션",
    "454910": "두산로보틱스",    "462870": "시프트업", "035900":"JYP Ent"
}

ENGLISH_NAME_MAP = {
    "alteogen inc.": "알테오젠","amorepacific corp.": "아모레퍼시픽", "celltrion, inc.": "셀트리온", "cj cheiljedang corp.": "CJ제일제당",
    "d&d pharmatech inc.": "디앤디파마텍", "daewoo engineering & construction co.,ltd": "대우건설",
    "doosan enerbility co., ltd.": "두산에너빌리티", "doosan fuel cell co., ltd.": "두산퓨얼셀", "doosan robotics inc.": "두산로보틱스", "e-mart inc.": "이마트",
    "ecopro bm co.,ltd.": "에코프로비엠", "hana financial group inc.": "하나금융지주", "hanmi pharm. co., ltd.": "한미약품", "hanmi semiconductor co., ltd.": "한미반도체",
    "hanwha aerospace co., ltd.": "한화에어로스페이스", "hanwha ocean co., ltd.": "한화오션", "hanwha solutions corporation": "한화솔루션",    "hd hyundai electric co.,ltd": "HD현대일렉트릭",
    "hd hyundai heavy industries co.,ltd.": "HD현대중공업", "hmm co.,ltd": "HMM", "hybe co., ltd.": "하이브", "hyosung heavy industries corporation": "효성중공업",
    "hyundai autoever corporation.": "현대오토에버", "hyundai engineering & construction co.,ltd": "현대건설","hyundai glovis co., ltd.": "현대글로비스",    "hyundai mobis co.,ltd": "현대모비스",
    "hyundai motor co": "현대자동차","hyundai steel company": "현대제철","hyundai-rotem co.": "현대로템","kakao corp.": "카카오","kb financial group inc.": "KB금융", "kepco engineering & construction company, inc": "한전기술",
    "kia corporation": "기아","korea aerospace industries, ltd.": "한국항공우주", "korea zinc inc": "고려아연", "krafton, inc.": "크래프톤",
    "kt corporation": "케이티",  "lg cns co., ltd.": "LG씨엔에스", "lg energy solution, ltd.": "LG에너지솔루션", "lg h&h co., ltd.": "LG생활건강",
    "lg innotek co., ltd.": "LG이노텍", "lg uplus corp": "LG유플러스", "lig defense&aerospace co., ltd.": "LIG디펜스앤에어로스페이스",    "ls electric co., ltd": "엘에스일렉트릭",
    "meritz financial group inc.": "메리츠금융지주",  "mirae asset securities co.,ltd.": "미래에셋증권",
    "naver corporation": "NAVER",  "nc corporation": "NC", "oci holdings company ltd.": "OCI홀딩스", "pharmaresearch co., ltd.": "파마리서치",
    "posco holdings inc.": "POSCO홀딩스",  "rainbow robotics": "레인보우로보틱스",  "samsung biologics co.,ltd.": "삼성바이오로직스",  "samsung e&a co.,ltd": "삼성E&A",
    "samsung electro-mechanics co.,ltd": "삼성전기",    "samsung electronics co,.ltd": "삼성전자",
    "samsung fire & marine insurance co.,ltd": "삼성화재해상보험",    "samsung heavy industries co.,ltd": "삼성중공업",
    "samsung life insurance co., ltd": "삼성생명", "samsung sdi co.,ltd": "삼성SDI", "seah besteel holdings corporation": "세아베스틸지주",    "shift up corporation.": "시프트업",
    "shinhan financial group co.,ltd": "신한지주",  "sk hynix inc.": "SK하이닉스", "sk telecom co.,ltd": "SK텔레콤", "sm entertainment co., ltd.": "에스엠",
    "woori financial group inc.": "우리금융지주",    "woori technology, incorporation": "우리기술",   "yg entertainment inc.": "와이지엔터테인먼트"
}

# 통용 약칭 / 한글 음차는 공식 데이터가 없어 사람이 관찰하며 계속 보강해야 하는 영역.
# 여기에 계속 추가해나가면 됨 (런타임 속도에는 영향 없음 — 그냥 dict 항목 추가일 뿐)
CORP_ALIASES = {
    "네이버": "NAVER",
    "포스코": "POSCO홀딩스",
    "하이닉스": "SK하이닉스",
    "에스케이하이닉스": "SK하이닉스",
    "엘지에너지솔루션": "LG에너지솔루션",
    "기아자동차": "기아",

    # 계속 추가...
}


def extract_target_corp(query, corp_names, aliases):
    """질문 문장에서 실제 metadata에 존재하는 기업명을 찾아 반환합니다.
    지원 표기: DART 공식 사명 > 종목코드 > 영문명 > 통용 약칭/한글음차 순으로 확인"""
    # ① DART 공식 사명 (가장 정확 - metadata의 corp_name과 직접 일치)
    for c in corp_names:
        if c and c in query:
            return c

    # ② 🔧 [수정 25] 종목코드 (6자리 숫자) - 정규식으로 추출 후 dict 조회
    code_match = re.search(r'\b(\d{6})\b', query)
    if code_match and code_match.group(1) in STOCK_CODE_MAP:
        return STOCK_CODE_MAP[code_match.group(1)]

    # ③ 🔧 [수정 25] 영문명 - 대소문자 무시하고 부분 일치
    query_lower = query.lower()
    for eng_name, kor_name in ENGLISH_NAME_MAP.items():
        if eng_name in query_lower:
            return kor_name

    # ④ 통용 약칭 / 한글 음차
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


# ==========================================
# 🔧 [수정 23] 신규 추가 - 자주 나오는 주제는 규칙으로 섹션을 고정
# 문제: "투자 계획" 질문은 항상 "원재료 및 생산설비"(설비투자)와
#      "주요계약 및 연구개발활동"(R&D) 두 섹션을 같이 봐야 완전한 답이 되는데,
#      LLM에게 "관련 섹션을 최대 3개 고르라"고 맡기면 매번 판단이 조금씩 달라져서
#      (비결정적) 둘 중 하나만 고르고 하나는 빠뜨리는 경우가 반복됨
# 해결: "투자"처럼 이미 정답 패턴이 확인된 반복 주제는 LLM 호출 없이
#      규칙으로 관련 섹션을 전부 확정. 이러면 매번 100% 동일하고 빠짐없이 잡힘.
#      규칙에 없는 새로운/애매한 주제만 기존처럼 LLM에게 맡김 (select_relevant_sections)
# ==========================================
UNINFORMATIVE_HEADINGS = {
    '사업보고서', '반기보고서', '분기보고서', '1분기보고서', '3분기보고서',
    '【 대표이사 등의 확인 】', 'I. 회사의 개요'
}

TOPIC_SECTION_HINTS = {
    ("연구개발", "r&d", "rnd", "연구개발비"): ["주요계약 및 연구개발", "연구개발실적", "연구개발", "매출 및 수주"],
    ("가이던스", "전망", "정정", "기재정정", "최근 공시", "수시공시"): ["사업의 개요", "주요계약 및 연구개발", "기타 참고사항", "기타 투자판단", "매출 및 수주"],
    ("배당", "배당금", "배당수익률", "주주환원", "자사주", "자기주식"): ["배당에 관한 사항", "연결재무제표 주석", "재무제표 주석", "요약재무정보", "주주"],
    ("당기순이익", "순이익", "영업이익", "손익", "재무제표", "증감률"): ["요약재무정보", "연결재무제표", "재무제표"],
    ("매출", "실적"): ["매출 및 수주", "매출실적", "매출 실적"],
    ("위험", "리스크"): ["위험관리", "파생거래", "이사의 경영진단"],
}

EQUITY_INVESTMENT_SIGNAL_KEYWORDS = ["지분", "인수", "M&A", "합병", "출자", "종속기업", "관계기업", "매각", "지분율"]
CAPEX_RND_SECTION_KEYWORDS = ["원재료 및 생산설비", "생산설비", "설비투자", "시설투자", "주요계약 및 연구개발", "연구개발"]
EQUITY_SECTION_KEYWORDS = ["타법인 출자", "종속기업", "관계기업", "지분법", "특수관계자", "연결대상"]


def rule_based_section_hint(query, available_sections):
    """알려진 주제 키워드가 질문에 있으면, 관련 섹션을 규칙으로 바로 확정합니다. (복합 키워드 지원)"""
    query_lower = query.lower()
    matched = set()

    if "투자" in query_lower:
        if any(kw in query_lower for kw in EQUITY_INVESTMENT_SIGNAL_KEYWORDS):
            matched |= {s for s in available_sections for kw in EQUITY_SECTION_KEYWORDS if kw in s}
        else:
            matched |= {s for s in available_sections for kw in CAPEX_RND_SECTION_KEYWORDS if kw in s}

    for topic_keys, keywords in TOPIC_SECTION_HINTS.items():
        if any(tk in query_lower for tk in topic_keys):
            matched |= {s for s in available_sections for kw in keywords if kw in s}

    return list(matched) if matched else None


def select_relevant_sections(query, available_sections):
    """문서의 목차 리스트 중, 질문에 답하기 위해 봐야 할 섹션을 LLM으로 선택합니다."""
    if not available_sections:
        return []

    meaningful_sections = [s for s in available_sections if s.strip() not in UNINFORMATIVE_HEADINGS]
    candidates = meaningful_sections if meaningful_sections else list(available_sections)

    if len(candidates) <= 1:
        return candidates

    sections_list_str = "\n".join(f"- {s}" for s in sorted(candidates))
    system_prompt = f"""당신은 기업 공시 보고서의 목차 중 질문과 가장 관련 있는 섹션을 고르는 전문 어시스턴트입니다.
아래는 이 보고서의 전체 목차 목록입니다:
{sections_list_str}

사용자 질문에 답하기 위해 반드시 확인해야 하는 가장 적절한 핵심 섹션을 목록에서 1~3개 고르세요.
단순 '사업보고서', '반기보고서', '대표이사 등의 확인' 같은 상위 껍데기 목차는 절대 선택하지 마세요.

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
            print(f"⚠️ [섹션 선택 API 오류] 상태 코드: {res.status_code}")
            return []
        result = res.json()
        content = result.get('result', {}).get('message', {}).get('content', '').strip()
        picked = [s.strip() for s in content.split(',') if s.strip()]
        valid = [s for s in picked if s in available_sections and s not in UNINFORMATIVE_HEADINGS]
        if not valid:
            for p in picked:
                for s in available_sections:
                    if p and (p in s or s in p) and s not in UNINFORMATIVE_HEADINGS:
                        valid.append(s)
        return list(dict.fromkeys(valid))
    except Exception as e:
        print(f"⚠️ [섹션 선택 예외] {e}")
        return []


def extract_target_year(query):
    match = re.search(r'(20\d{2})년', query)
    if match:
        return match.group(1)
    current_year = datetime.now().year
    if any(kw in query for kw in ["재작년"]):
        return str(current_year - 2)
    if any(kw in query for kw in ["작년", "전년도"]):
        return str(current_year - 1)
    if any(kw in query for kw in ["올해", "이번 년도", "이번년도", "금년"]):
        return str(current_year)
    return None

# 🔧 [수정 8] 신규 추가 - 질문에 포함된 보고서 종류(분기/반기/사업보고서) 추출
# 예: "2026년 1분기 분기보고서" -> "분기보고서"로 매칭되도록
# 🔧 [수정 20] 분기 → 문서종류 매핑을 실제 공시 제출 구조에 맞게 수정
#     문제: "2분기"를 "분기보고서"로 매핑했었는데, 한국 상장사는 2분기 전용 분기보고서를
#          제출하지 않음. 실제로는 아래처럼 제출됨:
#            1분기 -> 분기보고서 (1~3월)
#            2분기 -> 반기보고서 (1~6월 누적, "분기보고서"가 아님!)
#            3분기 -> 분기보고서 (1~9월 누적)
#            4분기 -> 별도 문서 없음, 사업보고서(연간)에 포함됨
#     기존 매핑대로면 "2분기" 질문이 실제 정답 문서(반기보고서)를 검색 대상에서
#     제외시켜버려서, 엉뚱하게 1분기/3분기 자료로 답변하는 사고가 발생했음
REPORT_TYPE_KEYWORDS = {
    "분기보고서": ["분기보고서", "1분기", "3분기", "분기 실적"],
    "반기보고서": ["반기보고서", "2분기", "반기 실적", "상반기"],
    "사업보고서": ["사업보고서", "4분기", "연간 실적", "연간보고서"],
}

def extract_target_report_type(query):
    for report_type, keywords in REPORT_TYPE_KEYWORDS.items():
        if any(kw in query for kw in keywords):
            return report_type
    # 분기/반기 언급 없이 연도(예: 2024년, 작년)만 있는 연간 실적 질의는 기본적으로 '사업보고서'를 우선 대상으로 설정
    if extract_target_year(query):
        return "사업보고서"
    return None

# ==========================================
# 💡 [신규 추가] 질문 재작성 (Query Rewriting) 함수
# 🔧 [수정 18] 규칙 기반 선처리 추가
#     문제: 기존엔 상대시점 표현이 있든 없든 매 질문마다 LLM API를 호출했음
#          ("2026년 1분기 보고서"처럼 이미 절대연도인 질문도 예외 없이 호출)
#     해결: "올해/작년/내년" 등 상대시점 키워드가 실제로 있는지 정규식으로 먼저 확인.
#          없으면 LLM 호출 없이 원본 질문을 그대로 반환 -> 불필요한 지연/비용 제거
#          있을 때만 LLM을 호출해 정확한 연도 계산(윤년 등 특수 로직 없이 문맥 이해 필요한 부분)을 맡김
# ==========================================
RELATIVE_TIME_KEYWORDS = ["올해", "이번 년도", "이번년도", "작년", "전년도", "재작년", "내년", "금년"]

def has_relative_time_expression(query):
    return any(kw in query for kw in RELATIVE_TIME_KEYWORDS)


def rewrite_query(original_query):
    """
    사용자 질문에 포함된 상대적 시간(올해, 작년, 재작년 등)을 절대 연도로 즉시 정확하게 변환합니다.
    """
    if not has_relative_time_expression(original_query):
        return original_query

    current_year = datetime.now().year
    converted = original_query
    
    # 1. 재작년 -> 2024년
    converted = re.sub(r'재작년(?:도)?', f'{current_year - 2}년', converted)
    # 2. 작년 / 전년도 -> 2025년
    converted = re.sub(r'(?:작년|전년도)', f'{current_year - 1}년', converted)
    # 3. 올해 / 이번 년도 / 이번년도 / 금년 -> 2026년
    converted = re.sub(r'(?:올해|이번\s*년도|금년)', f'{current_year}년', converted)
    # 4. 내년 -> 2027년
    converted = re.sub(r'내년', f'{current_year + 1}년', converted)
    
    return converted

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

    [중요 규칙: 부분 공시 및 미공시 항목 처리]
    - 질문에서 2가지 이상을 질의했으나 일부 항목만 공시된 경우(예: 북미 매출액은 있으나 지역별 영업이익은 미공시):
      확인 가능한 항목(예: 북미 매출액 83조 4,448억 원)을 우선 구체적인 수치와 단위(조/억원)로 정확하게 답변하고, 미공시된 세부 항목은 "다만 지역별 영업이익은 연결재무제표 주석(지역별 정보)상 별도로 분리 공시되지 않았습니다."와 같이 명확히 사실대로 안내하세요. 절대 전체를 정보 없음으로 거절하지 마세요.
    - [참고 자료] 전체가 질문 주제와 완전히 무관할 때만 "검색된 자료에서 요청하신 정보에 대한 구체적인 내용을 찾을 수 없습니다."라고 안내하세요.

    [답변 템플릿]
    {기업명} {연도/보고서명} 기준 주요 {질문 주제}:
    1. {핵심 항목1} — {수치 및 간결한 증감 요약}
    2. {핵심 항목2} — {수치 및 간결한 증감 요약}
    근거: 접수번호 {접수번호}, {목차명} / {목차명2}
    
    [지주회사 및 R&D/특수 공시 안내 규칙]
    - 지주회사(예: 세아베스틸지주 등)의 경우: 자체 공시상 별도 연구개발(R&D) 활동/비용이 기재되지 않거나 '해당사항 없음'으로 처리된 경우, 단순히 정보가 없다고 하지 말고 "지주회사 특성상 별도의 자체 연구개발(R&D) 조직 및 비용 지출이 없으며(해당사항 없음), 자회사 관리 및 배당/용역 중심의 영업수익을 창출하고 있습니다."와 같이 지주회사의 공시 사실과 사업 특성을 명확하게 설명하세요.

    [작성 예시]
    KB금융 2025년 사업보고서 기준 주요 실적:
    1. 당기순이익 — 5조 8,332억 원 (전년 대비 7,550억 원 증가)
    2. 주요 계열사 실적 — 국민은행 3조 852억 원, KB손해보험 7,780억 원, KB증권 6,740억 원 등
    근거: 접수번호 20260619000667, II. 사업의 내용 - 마. 그룹 영업실적
    
    [중요 규칙: 정정공시 처리]
    1. 검색된 [참고 자료]에 과거 문서(정정 전)와 최신 문서(정정 후) 내용이 모두 포함되어 있다면, 최신 버전의 수치와 증감률을 우선적으로 반영하세요.
    2. 최신 문서(is_latest가 true인 문서)의 내용을 최종 사실로 간주하세요.

    [중요 규칙: 누적치 vs 단일분기 구분]
    🔧 [수정 21] 신규 추가
    한국 공시 보고서의 매출/이익 수치는 대부분 "누적 기준"으로 표기됩니다.
    - 반기보고서의 수치 = 1~6월 누적 (2분기 단독 실적이 아님)
    - 3분기보고서의 수치 = 1~9월 누적 (3분기 단독 실적이 아님)
    사용자가 "2분기만", "3분기만"처럼 단일 분기 실적을 명시적으로 물었는데 자료에 누적치만 있다면,
    절대 누적치를 단일 분기 실적인 것처럼 제시하지 마세요. 반드시 "이 수치는 O월~O월 누적 기준입니다"라고
    명시하고, 가능하다면 [참고 자료]에 있는 직전 기간 누적치를 빼서 단일 분기 값을 계산해 보여주되
    "(직접 계산한 추정치)"라고 표시하세요. 계산할 재료가 없다면 누적치 그대로 제시하며 기준을 명확히 밝히세요.
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
def fast_rank_chunks(query, chunks, top_k=20):
    """수천~수만 개 청크가 있을 때, 조사 제거 및 복합 키워드 교차 매칭으로 0.05초 내에 핵심 표/데이터 청크를 정확히 선별합니다."""
    if len(chunks) <= top_k:
        return chunks

    raw_tokens = re.findall(r'[a-zA-Z0-9가-힣]+', query)
    stopwords = {
        '기준', '알려줘', '알려주세요', '얼마야', '어떻게', '대한', '관한', '에서', '으로', '보고서',
        '기준으로', '요약해', '전년', '대비', '올해', '작년', '재작년', '사업보고서', '분기보고서', '반기보고서',
        '2023년', '2024년', '2025년', '2026년', '현대자동차', '현대차', '카카오', '삼성전자', '포스코', '네이버'
    }

    keywords = set()
    for tok in raw_tokens:
        clean = re.sub(r'(?:과|와|을|를|의|은|는|이|가|에|로|으로|에서)$', '', tok)
        if len(clean) >= 2 and clean not in stopwords:
            keywords.add(clean)
        if len(tok) >= 2 and tok not in stopwords:
            keywords.add(tok)

    if not keywords:
        return chunks[:top_k]

    # 각 키워드의 문서 내 출현 빈도(문서 빈도)를 측정하여 희소 고유명사(예: '북미', '연구개발')에 초고가중치 부여
    kw_doc_counts = {kw: 0 for kw in keywords}
    for c in chunks:
        combined = c.get('heading_path', '') + ' ' + c.get('text', '')
        for kw in keywords:
            if kw in combined:
                kw_doc_counts[kw] += 1

    total_docs = len(chunks)
    kw_weights = {
        kw: max(2.0, 50.0 / (kw_doc_counts[kw] + 1)) if kw_doc_counts[kw] < total_docs * 0.05 else 1.0
        for kw in keywords
    }

    scored = []
    for c in chunks:
        h = c.get('heading_path', '')
        t = c.get('text', '')
        combined = h + ' ' + t
        matched_kws = [kw for kw in keywords if kw in combined]
        if matched_kws:
            # 매칭된 키워드 가중치 합산 + 다중 키워드 동시 출현 보너스
            s = sum(kw_weights[kw] for kw in matched_kws) * (len(matched_kws) ** 1.5)
            if any(kw in h for kw in matched_kws):
                s *= 2.0
            scored.append((s, c))

    if scored:
        scored.sort(key=lambda x: x[0], reverse=True)
        candidates = [c for s, c in scored[:100]]
    else:
        candidates = chunks[:100]

    if len(candidates) <= top_k:
        return candidates

    # 2단계: 선별된 후보에 대해서만 정밀 임베딩 코사인 유사도 계산 (0.05초 소요)
    q_vec = model.encode([query]).astype('float32')
    sample_texts = [(c.get('heading_path', '') + ' ' + c.get('text', ''))[:400] for c in candidates]
    c_vecs = model.encode(sample_texts, batch_size=64, show_progress_bar=False).astype('float32')

    q_norm = q_vec / (np.linalg.norm(q_vec, axis=1, keepdims=True) + 1e-9)
    c_norm = c_vecs / (np.linalg.norm(c_vecs, axis=1, keepdims=True) + 1e-9)
    scores = np.dot(c_norm, q_norm.T).squeeze()

    top_indices = np.argsort(scores)[::-1][:top_k]
    return [candidates[i] for i in top_indices]


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

            # 🔧 [수정 23] 규칙 기반 힌트를 먼저 시도 - 성공하면 LLM 호출 자체를 생략
            target_sections = rule_based_section_hint(search_query, available_sections)
            if target_sections:
                print(f"📂 [규칙 기반] 선택된 섹션: {target_sections}")
            else:
                # 규칙에 없는 주제라면 기존처럼 LLM에게 맡김
                target_sections = select_relevant_sections(search_query, available_sections)
                if target_sections:
                    print(f"📂 [LLM 선택] 선택된 섹션: {target_sections}")

    # 🔧 [수정 14] 신규 추가 - "섹션 직접 조회" 경로
    #     문제: 섹션까지 특정됐어도, 그 섹션의 청크가 벡터검색 top-300 순위 안에
    #          다 들어온다는 보장은 없음 (일부만 걸릴 수도 있음 -> 답변 누락 위험)
    #     해결: target_doc_ids + target_sections가 모두 확정되면, 벡터검색 없이
    #          메타데이터에서 해당 문서+섹션 청크를 통째로 직접 긁어옴 (정확도/재현율 100%)
    #     실패(섹션 선택이 안 됐거나 매칭 결과가 없음)하면 기존 벡터검색 경로로 자동 폴백
    retrieved_texts = []
    filtered_out_texts = []
    MAX_CONTEXT_CHUNKS = 20

    direct_chunks = []
    if target_sections and target_doc_ids:
        direct_chunks = [
            c for c in metadata
            if c.get('doc_id') in target_doc_ids and c.get('heading_path') in target_sections
        ]
        if direct_chunks:
            print(f"📌 섹션 직접 조회로 {len(direct_chunks)}개 청크를 찾았습니다.")
            if len(direct_chunks) > MAX_CONTEXT_CHUNKS:
                print(f"🎯 후보 청크({len(direct_chunks)}개)가 많아 고속 2단계 선별(키워드+유사도)로 상위 {MAX_CONTEXT_CHUNKS}개를 선별합니다...")
                direct_chunks = fast_rank_chunks(search_query, direct_chunks, MAX_CONTEXT_CHUNKS)

    # 🔧 [보완] 목차 누락/변형(예: 현대차, NAVER)으로 섹션 조회가 1개 이하인 경우 -> 해당 문서 전체 청크 대상 정밀 검색 수행
    if target_doc_ids and len(direct_chunks) <= 1:
        doc_chunks = [c for c in metadata if c.get('doc_id') in target_doc_ids]
        if doc_chunks and len(doc_chunks) > 1:
            print(f"🎯 문서 내 세부 섹션 미식별로 전체 {len(doc_chunks)}개 청크 대상 고속 정밀 검색을 수행합니다...")
            direct_chunks = fast_rank_chunks(search_query, doc_chunks, MAX_CONTEXT_CHUNKS)

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
