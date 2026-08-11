"""
2단계 (공시문서 선택): 질문에서 뽑아낸 '기업명 / 연도 / 분기 / 문서유형'으로
manifest.jsonl에서 정확한 문서(rcept_no, file_path)를 찾아주는 모듈.

전제:
    - 1단계(의도 파악)에서 이미 corp_query(회사명), year, quarter 등을
      추출했다고 가정한다. (예: "삼성전자 2026년 1분기 투자 계획" ->
      corp_query="삼성전자", year=2026, quarter=1)

사용 예:
    from build_context import CorpusIndex

    idx = CorpusIndex(
        universe_csv="universe.csv",
        manifest_jsonl="manifest.jsonl",
    )

    docs = idx.find_documents(corp_query="현대차", year=2023, quarter=1)
    for d in docs:
        print(d["report_nm"], d["file_path"])
"""

from pathlib import Path
import pandas as pd


class CorpusIndex:
    def __init__(self, universe_csv: str, manifest_jsonl: str):
        self.universe = pd.read_csv(
            universe_csv, dtype={"corp_code": str, "stock_code": str}
        )
        self.manifest = pd.read_json(
            manifest_jsonl, lines=True, dtype={"corp_code": str, "stock_code": str}
        )

    # ---------- 1. 회사명 별칭 -> corp_code ----------
    def resolve_corp_code(self, corp_query: str) -> str | None:
        """
        사용자가 부른 이름(통용명, 약칭 등)을 universe.csv의 corp_name/listed_name과
        대조해서 corp_code를 찾는다. 정확히 일치하는 게 없으면 부분 일치로 재시도.
        """
        q = corp_query.strip()

        # 정확히 일치 (corp_name 또는 listed_name)
        exact = self.universe[
            (self.universe["corp_name"] == q) | (self.universe["listed_name"] == q)
        ]
        if len(exact) == 1:
            return exact.iloc[0]["corp_code"]

        # 부분 일치 (예: "현대차" -> "현대자동차" 포함 매칭)
        partial = self.universe[
            self.universe["corp_name"].str.contains(q, na=False)
            | self.universe["listed_name"].str.contains(q, na=False)
        ]
        if len(partial) >= 1:
            # 시가총액이 가장 큰 후보를 우선 (동명이인·유사기업 방지)
            return partial.sort_values("market_cap", ascending=False).iloc[0]["corp_code"]

        return None

    # ---------- 2. 조건에 맞는 문서 검색 ----------
    def find_documents(
        self,
        corp_query: str,
        year: int | None = None,
        quarter: int | None = None,   # 1~4, 반기는 quarter=2 -> base_month=6 로 매핑됨
        doc_group: str = "periodic",  # periodic | major | exchange | holding
        doc_subtype: str | None = None,  # annual | half | quarter 등
        include_corrections: bool = False,
    ) -> list[dict]:
        corp_code = self.resolve_corp_code(corp_query)
        if corp_code is None:
            return []

        df = self.manifest[self.manifest["corp_code"] == corp_code]
        df = df[df["doc_group"] == doc_group]

        if not include_corrections:
            df = df[~df["is_correction"]]

        if year is not None:
            df = df[df["base_year"] == year]

        if quarter is not None:
            # 분기보고서 base_month: 3,6,9 / 반기 base_month: 6 / 사업보고서(연간) base_month: 12
            quarter_to_month = {1: 3, 2: 6, 3: 9, 4: 12}
            df = df[df["base_month"] == quarter_to_month.get(quarter, quarter)]

        if doc_subtype is not None:
            df = df[df["doc_subtype"] == doc_subtype]

        df = df.sort_values("rcept_dt", ascending=False)
        return df.to_dict(orient="records")

    # ---------- 3. XML 실제 경로 얻기 ----------
    def get_xml_path(self, doc_record: dict, corpus_root: str) -> Path:
        """
        find_documents()가 반환한 레코드 하나를 받아 실제 XML 파일 경로를 반환한다.
        file_path는 폴더까지만 가리키므로, 그 안의 .xml 파일을 찾는다.
        """
        folder = Path(corpus_root) / doc_record["file_path"]
        xml_files = list(folder.glob("*.xml"))
        if not xml_files:
            raise FileNotFoundError(f"XML 파일을 찾을 수 없음: {folder}")
        return xml_files[0]


if __name__ == "__main__":
    # 간단한 동작 확인
    idx = CorpusIndex(
        universe_csv="/mnt/user-data/uploads/universe.csv",
        manifest_jsonl="/mnt/user-data/uploads/manifest.jsonl",
    )

    print("=== 테스트 1: 삼성전자 2023년 1분기 분기보고서 ===")
    docs = idx.find_documents(corp_query="삼성전자", year=2023, quarter=1)
    for d in docs:
        print(d["report_nm"], "|", d["rcept_no"], "|", d["file_path"])

    print("\n=== 테스트 2: 별칭 매칭 - '현대차' -> 현대자동차 ===")
    code = idx.resolve_corp_code("현대차")
    print("resolve_corp_code('현대차') ->", code)
    docs = idx.find_documents(corp_query="현대차", year=2024, quarter=2)
    for d in docs:
        print(d["report_nm"], "|", d["rcept_no"], "|", d["file_path"])
