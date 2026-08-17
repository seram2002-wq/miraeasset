# batch_parse_exchange.py

import json
import re
import sys
from pathlib import Path

import pandas as pd

from parse_exchange_xml import parse_exchange_xml


def make_chunks(doc):
    """
    파싱된 거래소공시 1건을 section별 chunk로 변환
    """

    text = doc.get("text", "").strip()

    if not text:
        return []

    # ## 1. ..., ## 2. ... 기준으로 분리
    sections = re.split(
        r"\n(?=##\s+)",
        text
    )

    chunks = []

    for idx, section in enumerate(sections):

        section = section.strip()

        # section이 아닌 제목 부분은 제외
        if not section.startswith("## "):
            continue

        lines = section.splitlines()

        # "## 2. 계약내역"
        section_name = re.sub(
            r"^##\s+",
            "",
            lines[0]
        ).strip()

        # section 번호 추출
        match = re.match(
            r"^(\d+)\.",
            section_name
        )

        section_code = (
            match.group(1)
            if match
            else str(idx)
        )

        body = "\n".join(
            lines[1:]
        ).strip()

        # 접수일 YYYY-MM-DD
        rcept_dt = str(
            doc.get("rcept_dt", "")
        )

        if len(rcept_dt) == 8:
            date = (
                f"{rcept_dt[:4]}-"
                f"{rcept_dt[4:6]}-"
                f"{rcept_dt[6:]}"
            )
        else:
            date = rcept_dt

        # 세람 결과와 비슷한 text 구조
        chunk_text = (
            f"[{doc['corp_name']}] "
            f"{doc.get('report_nm', '')} "
            f"({date})\n"
            f"섹션: {section_name}\n"
            f"{body}"
        ).strip()

        chunk = {
            "chunk_id": (
                f"{doc['doc_id']}-"
                f"{section_code}-0"
            ),

            "doc_id": doc["doc_id"],

            "corp_code": doc["corp_code"],
            "corp_name": doc["corp_name"],
            "stock_code": doc["stock_code"],

            "industry": doc.get("industry"),
            "sector": doc.get("sector"),

            "doc_group": doc["doc_group"],
            "doc_subtype": doc.get("doc_subtype"),
            "report_nm": doc.get("report_nm"),

            "rcept_no": str(
                doc["rcept_no"]
            ),

            "rcept_dt": date,

            "is_correction": doc.get(
                "is_correction",
                False
            ),

            "section_path": section_name,
            "section_code": section_code,

            "text": chunk_text
        }

        chunks.append(chunk)

    return chunks


def main():

    if len(sys.argv) < 3:

        print(
            "사용법:"
        )

        print(
            "python batch_parse_exchange.py "
            "<corpus_root> <output.jsonl>"
        )

        print(
            "예:"
        )

        print(
            "python batch_parse_exchange.py "
            "corpus exchange.jsonl"
        )

        return

    corpus_root = Path(sys.argv[1])
    output_file = Path(sys.argv[2])

    universe_file = (
        corpus_root / "universe.csv"
    )

    manifest_file = (
        corpus_root / "manifest.jsonl"
    )

    # --------------------------------------------------
    # universe.csv 확인
    # --------------------------------------------------

    if not universe_file.exists():

        raise FileNotFoundError(
            f"universe.csv를 찾을 수 없습니다: "
            f"{universe_file}"
        )

    # --------------------------------------------------
    # manifest.jsonl 확인
    # --------------------------------------------------

    if not manifest_file.exists():

        raise FileNotFoundError(
            f"manifest.jsonl을 찾을 수 없습니다: "
            f"{manifest_file}"
        )

    # --------------------------------------------------
    # universe 읽기
    # --------------------------------------------------

    universe = pd.read_csv(
        universe_file,
        encoding="utf-8-sig",
        dtype=str
    )

    # --------------------------------------------------
    # manifest 읽기
    # --------------------------------------------------

    manifest = pd.read_json(
        manifest_file,
        lines=True,
        dtype=False
    )

    # 거래소공시만 선택
    exchange_docs = manifest[
        manifest["doc_group"]
        == "exchange"
    ]

    print(
        f"[INFO] 거래소공시 대상: "
        f"{len(exchange_docs)}건"
    )

    success = 0
    failed = 0
    total_chunks = 0

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------
    # JSONL 저장
    # --------------------------------------------------

    with output_file.open(
        "w",
        encoding="utf-8"
    ) as fout:

        for i, (_, row) in enumerate(
            exchange_docs.iterrows(),
            start=1
        ):

            doc_id = str(
                row["doc_id"]
            )

            corp_name = str(
                row["corp_name"]
            )

            rcept_no = str(
                row["rcept_no"]
            )

            try:

                # --------------------------------------------------
                # XML 경로
                # manifest의 file_path는 폴더까지
                # --------------------------------------------------

                folder = (
                    corpus_root
                    / str(row["file_path"])
                )

                xml_files = list(
                    folder.glob("*.xml")
                )

                if not xml_files:

                    raise FileNotFoundError(
                        f"XML 파일을 찾을 수 없음: "
                        f"{folder}"
                    )

                xml_path = xml_files[0]

                # --------------------------------------------------
                # XML parsing
                # --------------------------------------------------

                parsed = parse_exchange_xml(
                    xml_path
                )

                # --------------------------------------------------
                # metadata 합치기
                # --------------------------------------------------

                doc = {
                    "doc_id": doc_id,

                    "corp_code": str(
                        row["corp_code"]
                    ),

                    "corp_name": corp_name,

                    "listed_name": str(
                        row["listed_name"]
                    ),

                    "stock_code": str(
                        row["stock_code"]
                    ),

                    "industry": str(
                        row["industry"]
                    ),

                    "sector": str(
                        row["sector"]
                    ),

                    "doc_group": "exchange",

                    "doc_subtype": str(
                        row["doc_subtype"]
                    ),

                    "report_nm": str(
                        row["report_nm"]
                    ),

                    "rcept_no": rcept_no,

                    "rcept_dt": str(
                        row["rcept_dt"]
                    ),

                    "is_correction": (
                        bool(row["is_correction"])
                        if not pd.isna(
                            row["is_correction"]
                        )
                        else False
                    ),

                    "title": parsed["title"],
                    "text": parsed["text"]
                }

                # --------------------------------------------------
                # chunk 생성
                # --------------------------------------------------

                chunks = make_chunks(doc)

                if not chunks:
                    print(
                        f"[SKIP] {i}/"
                        f"{len(exchange_docs)} "
                        f"{corp_name} / "
                        f"{rcept_no}: "
                        f"chunk 없음"
                    )

                    continue

                # --------------------------------------------------
                # JSONL에 바로 기록
                # --------------------------------------------------

                for chunk in chunks:

                    fout.write(
                        json.dumps(
                            chunk,
                            ensure_ascii=False
                        )
                        + "\n"
                    )

                    total_chunks += 1

                success += 1

                print(
                    f"[OK] {i}/"
                    f"{len(exchange_docs)} "
                    f"{corp_name} / "
                    f"{rcept_no} "
                    f"→ {len(chunks)} chunks"
                )

            except Exception as e:

                failed += 1

                print(
                    f"[ERROR] {i}/"
                    f"{len(exchange_docs)} "
                    f"{corp_name} / "
                    f"{rcept_no}: "
                    f"{e}"
                )

    print()
    print(
        "===== 처리 완료 ====="
    )
    print(
        f"전체 공시: "
        f"{len(exchange_docs)}"
    )
    print(
        f"성공 공시: "
        f"{success}"
    )
    print(
        f"실패 공시: "
        f"{failed}"
    )
    print(
        f"생성 chunk: "
        f"{total_chunks}"
    )
    print(
        f"출력: "
        f"{output_file}"
    )


if __name__ == "__main__":
    main()