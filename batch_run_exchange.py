# -*- coding: utf-8 -*-
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def load_manifest_all(manifest_path):
    table = {}
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return table
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rcept_no = d.get("rcept_no")
            if rcept_no:
                table[str(rcept_no)] = d
    return table


def process_one(xml_path_str, meta, out_dir_str):
    import dart_exchange_chunker as dec

    xml_path = Path(xml_path_str)
    out_dir = Path(out_dir_str)
    rcept_no = xml_path.stem

    result = {
        "rcept_no": rcept_no,
        "xml_path": str(xml_path),
        "status": "ok",
        "n_chunks": 0,
        "manifest_matched": meta is not None,
        "error": None,
    }

    try:
        if meta is None:
            meta = {}

        doc_name, company_name, sections = dec.extract_exchange_blocks(xml_path)

        doc_group_map = {
            "periodic": "정기공시",
            "major": "주요사항보고서",
            "exchange": "거래소공시",
            "holding": "지분공시",
        }
        file_path_str = meta.get("file_path", "")
        base_meta = {
            "source_doc_type": doc_group_map.get(meta.get("doc_group", "exchange"), "거래소공시"),
            "doc_subtype": meta.get("doc_subtype", ""),
            "corp_name": meta.get("corp_name", company_name),
            "corp_code": meta.get("corp_code", ""),
            "stock_code": meta.get("stock_code", ""),
            "industry": meta.get("industry", ""),
            "sector": meta.get("sector", ""),
            "rcept_no": rcept_no,
            "rcept_dt": meta.get("rcept_dt", ""),
            "report_nm": meta.get("report_nm", doc_name),
            "is_correction": meta.get("is_correction", False),
            "source_file": f"corpus/{file_path_str}/{xml_path.name}",
        }

        chunks = dec.build_chunks(sections, base_meta)
        result["n_chunks"] = len(chunks)

        chunks_path = out_dir / f"{rcept_no}.chunks.jsonl"
        with open(chunks_path, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

    except Exception as e:
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {e}"

    return result


def merge_chunks(out_dir: Path, merged_name="_merged_all_chunks.jsonl"):
    out_dir = Path(out_dir)
    merged_path = out_dir / merged_name
    chunk_files = sorted(out_dir.glob("*.chunks.jsonl"))
    chunk_files = [p for p in chunk_files if p.name != merged_name]

    print(f"[MERGE] {len(chunk_files)}개 청크 파일을 {merged_path}로 병합 중...")
    total_lines = 0
    with open(merged_path, "w", encoding="utf-8") as fout:
        for p in chunk_files:
            with open(p, encoding="utf-8") as fin:
                for line in fin:
                    if line.strip():
                        fout.write(line)
                        total_lines += 1
    print(f"[MERGE] 완료! 총 {total_lines}개 청크 병합됨 -> {merged_path}")
    return merged_path, total_lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="corpus/raw/exchange")
    ap.add_argument("--manifest", default="corpus/manifest.jsonl")
    ap.add_argument("--out-dir", default="out/exchange")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INIT] manifest 로드: {args.manifest}")
    manifest_table = load_manifest_all(args.manifest)

    xml_files = sorted(raw_dir.glob("**/*.xml"))
    if args.limit:
        xml_files = xml_files[:args.limit]

    print(f"[INIT] 대상 XML 파일: 총 {len(xml_files)}개 (워커: {args.workers})")

    tasks = []
    skipped = 0
    for xp in xml_files:
        rcept_no = xp.stem
        if args.skip_existing and (out_dir / f"{rcept_no}.chunks.jsonl").exists():
            skipped += 1
            continue
        tasks.append((str(xp), manifest_table.get(rcept_no), str(out_dir)))

    print(f"[RUN] 처리할 작업: {len(tasks)}개 (이미 처리됨 건너뜀: {skipped}개)")

    start_t = time.time()
    results = []

    if args.workers <= 1:
        for i, t in enumerate(tasks, start=1):
            res = process_one(*t)
            results.append(res)
            if i % 100 == 0 or i == len(tasks):
                print(f"[{i}/{len(tasks)}] {res['rcept_no']} ({res['status']}, {res['n_chunks']} chunks)")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_one, *t): t[0] for t in tasks}
            for i, fut in enumerate(as_completed(futs), start=1):
                res = fut.result()
                results.append(res)
                if i % 200 == 0 or i == len(tasks):
                    print(f"[{i}/{len(tasks)}] 처리 중... (마지막: {res['rcept_no']} {res['status']})")

    elapsed = time.time() - start_t
    ok_cnt = sum(1 for r in results if r["status"] == "ok")
    err_cnt = sum(1 for r in results if r["status"] == "error")
    total_chunks = sum(r["n_chunks"] for r in results)

    print(f"\n===== 거래소공시 배치 완료 ({elapsed:.1f}초) =====")
    print(f"성공: {ok_cnt}건, 실패: {err_cnt}건, 생성 청크: {total_chunks}개")

    if args.merge:
        merge_chunks(out_dir)


if __name__ == "__main__":
    main()
