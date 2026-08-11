"""
raw/periodic/ 전체를 순회하며 모든 정기공시(사업·반기·분기보고서) XML을
섹션 단위로 파싱해서 하나의 JSONL 파일로 합친다.

전제:
    - 같은 폴더에 build_context.py, parse_dart_xml.py가 있어야 한다.
    - corpus_root 안에 manifest.jsonl, universe.csv, raw/ 가 들어있어야 한다.
      (즉 corpus_root = 압축 풀어놓은 데이터셋 최상위 폴더, 보통 바탕화면의 그 폴더)

사용법:
    python batch_parse_periodic.py <corpus_root> <출력.jsonl>

    예) python batch_parse_periodic.py . periodic_sections.jsonl
        (현재 폴더가 corpus_root인 경우)

출력 (JSONL, 한 줄 = 섹션 하나):
    {
      "corp_name": "CJ제일제당", "rcept_no": "20230515002270",
      "report_nm": "분기보고서 (2023.03)", "base_year": 2023, "base_month": 3,
      "section_id": "L-0-2-3-L1", "title": "3. 원재료 및 생산설비",
      "text": "...", "char_count": 4394
    }
"""

import sys
import json
from pathlib import Path

from build_context import CorpusIndex
from parse_dart_xml import parse_dart_xml


def batch_parse_periodic(corpus_root: str, output_jsonl: str):
    corpus_root = Path(corpus_root)
    idx = CorpusIndex(
        universe_csv=str(corpus_root / "universe.csv"),
        manifest_jsonl=str(corpus_root / "manifest.jsonl"),
    )

    # doc_group="periodic" 전체 (기업 조건 없이) — manifest를 직접 필터링
    docs = idx.manifest[
        (idx.manifest["doc_group"] == "periodic")
        & (idx.manifest["file_format"] == "xml")  # pdf+html 대체수집 3건 제외
    ].to_dict(orient="records")

    print(f"대상 문서 수: {len(docs)}")

    n_ok, n_fail = 0, 0
    with open(output_jsonl, "w", encoding="utf-8") as out_f:
        for i, doc in enumerate(docs, 1):
            try:
                xml_path = idx.get_xml_path(doc, corpus_root=str(corpus_root))
                parsed = parse_dart_xml(str(xml_path))

                for sec in parsed["sections"]:
                    if not sec["text"]:
                        continue  # 빈 섹션(제목만 있고 내용 없는 상위 챕터)은 제외
                    row = {
                        "corp_name": doc["corp_name"],
                        "rcept_no": doc["rcept_no"],
                        "report_nm": doc["report_nm"],
                        "base_year": doc["base_year"],
                        "base_month": doc["base_month"],
                        "section_id": sec["section_id"],
                        "title": sec["title"],
                        "text": sec["text"],
                        "char_count": sec["char_count"],
                    }
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_ok += 1
            except Exception as e:
                n_fail += 1
                print(f"  [실패] {doc['corp_name']} {doc['rcept_no']}: {e}")

            if i % 100 == 0:
                print(f"  진행: {i}/{len(docs)}")

    print(f"\n완료 — 성공 {n_ok}건 / 실패 {n_fail}건")
    print(f"저장 위치: {output_jsonl}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("사용법: python batch_parse_periodic.py <corpus_root> <출력.jsonl>")
        sys.exit(1)

    batch_parse_periodic(corpus_root=sys.argv[1], output_jsonl=sys.argv[2])
