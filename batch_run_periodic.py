"""
DART 정기공시 배치 실행기
================================================================
dart_periodic_chunker.py 의 핵심 함수(extract_blocks / render_markdown /
build_chunks)를 그대로 재사용해서, raw/periodic/ 하위의 모든 XML을 순회하며
{rcept_no}.md / {rcept_no}.chunks.jsonl 을 생성한다.

원본 스크립트의 main()을 파일마다 subprocess로 재호출하지 않는 이유:
  - main()은 manifest.jsonl 전체를 매번 처음부터 읽는다 (load_manifest_meta).
    70개사 x 1,054건 규모에서 이걸 1,054번 반복하면 I/O가 쓸데없이 커진다.
    -> 배치 스크립트는 manifest를 한 번만 읽어 dict로 올려두고 재사용한다.
  - 파일 하나가 깨져도(XML 파싱 실패, manifest 미매칭 등) 전체 배치가 죽으면 안 된다.
    -> 파일 단위로 try/except, 실패는 로그에 남기고 계속 진행.
  - 1,054건 처리 진행 상황을 볼 수 있어야 한다.
    -> 진행률 출력 + 실행 로그(run_log.jsonl) + 요약(summary.json) 생성.
  - 중간에 멈췄다 재실행할 수 있어야 한다.
    -> --skip-existing 으로 이미 만든 산출물은 건너뜀.

사용:
    # 순차 처리
    python3 batch_run_periodic.py --raw-dir raw/periodic --manifest manifest.jsonl --out-dir out/periodic

    # 병렬 처리 (프로세스 4개)
    python3 batch_run_periodic.py --raw-dir raw/periodic --manifest manifest.jsonl --out-dir out/periodic --workers 4

    # 중간 재실행 (이미 만든 산출물은 건너뜀) + 끝나고 전체 병합
    python3 batch_run_periodic.py --raw-dir raw/periodic --manifest manifest.jsonl --out-dir out/periodic --skip-existing --merge

    # 일부만 먼저 돌려보고 확인 (n=5)
    python3 batch_run_periodic.py --raw-dir raw/periodic --manifest manifest.jsonl --out-dir out/periodic --limit 5
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# dart_periodic_chunker.py 와 같은 폴더에 있다고 가정하고 import.
# (다른 폴더에 있으면 --chunker-path 로 지정)
sys.path.insert(0, str(Path(__file__).resolve().parent))


# ----------------------------------------------------------------------
# manifest 전체를 한 번만 읽어 dict로 변환 (rcept_no -> record)
# ----------------------------------------------------------------------
def load_manifest_all(manifest_path):
    meta_by_rcept = {}
    n_lines = 0
    n_bad = 0
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                n_bad += 1
                continue
            rcept_no = d.get("rcept_no")
            if rcept_no:
                meta_by_rcept[rcept_no] = d
    if n_bad:
        print(f"[경고] manifest.jsonl 에서 {n_bad}줄을 JSON 파싱 실패로 건너뜀", file=sys.stderr)
    return meta_by_rcept


# ----------------------------------------------------------------------
# XML 1건 처리 (dart_periodic_chunker의 main() 본문과 동일한 로직,
# 다만 manifest는 이미 로드된 dict에서 조회)
# ----------------------------------------------------------------------
def process_one(xml_path_str, meta, out_dir_str):
    """워커(프로세스풀)에서도 호출되므로 인자는 모두 pickle 가능한 기본 타입만 사용."""
    import dart_periodic_chunker as dpc  # 워커 프로세스마다 import (fork/spawn 모두 안전)

    xml_path = Path(xml_path_str)
    out_dir = Path(out_dir_str)
    rcept_no = xml_path.stem

    result = {
        "rcept_no": rcept_no,
        "xml_path": str(xml_path),
        "status": "ok",
        "n_blocks": 0,
        "n_chunks": 0,
        "manifest_matched": meta is not None,
        "empty_blocks": False,
        "error": None,
    }

    try:
        if meta is None:
            meta = {}

        doc_name, company_name, blocks = dpc.extract_blocks(xml_path)
        result["n_blocks"] = len(blocks)

        if len(blocks) == 0:
            # lxml은 recover=True 라서 심하게 깨진 XML도 예외 없이 "빈 문서"로
            # 통과시켜버린다. 이걸 그냥 성공으로 세면 손상 파일이 조용히 묻힌다.
            result["empty_blocks"] = True

        ##md = dpc.render_markdown(doc_name, company_name or meta.get("corp_name", ""), blocks)
        ##md_path = out_dir / f"{rcept_no}.md"
        ##md_path.write_text(md, encoding="utf-8")

        doc_group_map = {
            "periodic": "정기공시",
            "major": "주요사항보고서",
            "exchange": "거래소공시",
            "holding": "지분공시",
        }
        base_meta = {
            "source_doc_type": doc_group_map.get(meta.get("doc_group", "periodic"), "정기공시"),
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
            "base_year": meta.get("base_year"),
            "base_month": meta.get("base_month"),
            "source_file": meta.get("file_path", str(xml_path)) + f"/{xml_path.name}",
        }

        chunks = dpc.build_chunks(blocks, base_meta)
        result["n_chunks"] = len(chunks)

        chunks_path = out_dir / f"{rcept_no}.chunks.jsonl"
        with open(chunks_path, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

        if not meta:
            result["status"] = "ok_no_manifest"  # 처리는 됐지만 manifest 매칭 실패

    except Exception as e:
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"

    return result


# ----------------------------------------------------------------------
# 산출물 병합 (선택)
# ----------------------------------------------------------------------
def merge_chunks(out_dir, merged_name="_merged_all_chunks.jsonl"):
    out_dir = Path(out_dir)
    merged_path = out_dir / merged_name
    n = 0
    with open(merged_path, "w", encoding="utf-8") as out_f:
        for p in sorted(out_dir.glob("*.chunks.jsonl")):
            if p.name == merged_name:
                continue
            with open(p, encoding="utf-8") as in_f:
                for line in in_f:
                    line = line.rstrip("\n")
                    if line:
                        out_f.write(line + "\n")
                        n += 1
    return merged_path, n


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="raw/periodic/ 하위 XML 전체를 순회하며 청킹 배치 실행")
    ap.add_argument("--raw-dir", default="raw/periodic", help="XML이 들어있는 루트 폴더 (하위 폴더까지 재귀 탐색)")
    ap.add_argument("--manifest", default="manifest.jsonl", help="manifest.jsonl 경로")
    ap.add_argument("--out-dir", default="out/periodic", help="결과(.md/.chunks.jsonl) 저장 폴더")
    ap.add_argument("--workers", type=int, default=1, help="병렬 프로세스 수 (기본 1=순차)")
    ap.add_argument("--limit", type=int, default=None, help="테스트용: 앞에서 N건만 처리")
    ap.add_argument("--skip-existing", action="store_true",
                     help="{rcept_no}.chunks.jsonl 이 이미 있으면 건너뜀 (재실행/이어하기용)")
    ap.add_argument("--merge", action="store_true",
                     help="처리 끝난 뒤 out-dir 내 모든 *.chunks.jsonl 을 _merged_all_chunks.jsonl 로 합침")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        print(f"[오류] raw-dir 이 존재하지 않습니다: {raw_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[1/4] manifest 로딩: {args.manifest}")
    meta_by_rcept = load_manifest_all(args.manifest)
    print(f"      manifest 레코드 {len(meta_by_rcept)}건 로드")

    print(f"[2/4] XML 탐색: {raw_dir} (재귀)")
    xml_files = sorted(raw_dir.rglob("*.xml"))
    if args.limit:
        xml_files = xml_files[: args.limit]
    print(f"      대상 XML {len(xml_files)}건")

    if args.skip_existing:
        before = len(xml_files)
        xml_files = [
            p for p in xml_files
            if not (out_dir / f"{p.stem}.chunks.jsonl").exists()
        ]
        print(f"      --skip-existing: {before - len(xml_files)}건 건너뜀 (이미 산출물 존재), 남은 {len(xml_files)}건")

    if not xml_files:
        print("처리할 XML이 없습니다. 종료.")
        return

    tasks = [(str(p), meta_by_rcept.get(p.stem), str(out_dir)) for p in xml_files]

    print(f"[3/4] 처리 시작 (workers={args.workers})")
    t0 = time.time()
    results = []
    n_done = 0
    total = len(tasks)
    log_interval = max(1, total // 20)  # 대략 5%마다 진행 로그

    def report(res):
        nonlocal n_done
        n_done += 1
        if n_done % log_interval == 0 or n_done == total:
            elapsed = time.time() - t0
            rate = n_done / elapsed if elapsed > 0 else 0
            eta = (total - n_done) / rate if rate > 0 else float("inf")
            print(f"      {n_done}/{total} 완료 ({rate:.1f}건/초, 남은 예상 {eta:.0f}초)")

    if args.workers <= 1:
        for xml_str, meta, out_str in tasks:
            res = process_one(xml_str, meta, out_str)
            results.append(res)
            report(res)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(process_one, x, m, o) for x, m, o in tasks]
            for fut in as_completed(futures):
                res = fut.result()
                results.append(res)
                report(res)

    elapsed = time.time() - t0

    # ---- 요약 집계 ----
    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_ok_no_manifest = sum(1 for r in results if r["status"] == "ok_no_manifest")
    n_error = sum(1 for r in results if r["status"] == "error")
    n_empty = sum(1 for r in results if r.get("empty_blocks"))
    total_blocks = sum(r["n_blocks"] for r in results)
    total_chunks = sum(r["n_chunks"] for r in results)

    print(f"[4/4] 완료: {elapsed:.1f}초")
    print(f"      성공 {n_ok}건 / manifest 미매칭(처리는 됨) {n_ok_no_manifest}건 / 실패(예외) {n_error}건")
    if n_empty:
        print(f"      [주의] blocks=0 (사실상 빈 결과, XML 손상 가능성) {n_empty}건 "
              f"-> lxml recover=True 때문에 예외 없이 통과되므로 반드시 확인 필요")
    print(f"      총 blocks {total_blocks} -> 총 chunks {total_chunks}")

    # ---- 로그 파일 저장 ----
    run_log_path = out_dir / "run_log.jsonl"
    with open(run_log_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "raw_dir": str(raw_dir),
        "manifest": args.manifest,
        "out_dir": str(out_dir),
        "n_xml_total": total,
        "n_ok": n_ok,
        "n_ok_no_manifest": n_ok_no_manifest,
        "n_error": n_error,
        "n_empty_blocks": n_empty,
        "total_blocks": total_blocks,
        "total_chunks": total_chunks,
        "elapsed_sec": round(elapsed, 1),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"      실행 로그: {run_log_path}")
    print(f"      요약: {summary_path}")

    if n_error:
        failed_path = out_dir / "failed_files.txt"
        with open(failed_path, "w", encoding="utf-8") as f:
            for r in results:
                if r["status"] == "error":
                    f.write(f"{r['xml_path']}\n{r['error']}\n{'-'*60}\n")
        print(f"      [주의] 실패 {n_error}건 상세: {failed_path}")

    if n_empty:
        empty_path = out_dir / "empty_blocks_files.txt"
        with open(empty_path, "w", encoding="utf-8") as f:
            for r in results:
                if r.get("empty_blocks"):
                    f.write(f"{r['xml_path']}\n")
        print(f"      [주의] blocks=0 파일 목록: {empty_path}")

    if n_ok_no_manifest:
        print(f"      [주의] manifest에서 rcept_no를 못 찾은 파일 {n_ok_no_manifest}건 "
              f"-> run_log.jsonl 에서 manifest_matched=false 로 확인 가능")

    # ---- 병합 (선택) ----
    if args.merge:
        merged_path, n_lines = merge_chunks(out_dir)
        print(f"      병합 완료: {merged_path} ({n_lines}개 청크, 임베딩 단계 입력용)")


if __name__ == "__main__":
    main()
