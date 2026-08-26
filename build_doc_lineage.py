# -*- coding: utf-8 -*-
import argparse
import json
import os
import re
import sys
from pathlib import Path
from bs4 import BeautifulSoup
import pandas as pd
import warnings
from bs4 import XMLParsedAsHTMLWarning
warnings.filterwarnings('ignore', category=XMLParsedAsHTMLWarning)


def clean_report_title(title: str) -> str:
    if not title:
        return ''
    t = re.sub(r'\[(기재정정|첨부추가|정정|첨부정정)\]', '', str(title))
    t = re.sub(r'\(자율공시\)', '', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def extract_orig_rcept_dt_from_xml(corpus_root: Path, file_path: str) -> str | None:
    if not file_path:
        return None
    folder = corpus_root / str(file_path).replace('/', os.sep)
    if not folder.exists():
        return None
    xml_files = list(folder.glob('*.xml'))
    if not xml_files:
        return None
    try:
        content = xml_files[0].read_text(encoding='utf-8-sig', errors='replace')
        soup = BeautifulSoup(content, 'html.parser')
        for tr in soup.find_all('tr'):
            cells = [clean_report_title(td.get_text(' ', strip=True)) for td in tr.find_all(['td', 'th'])]
            if len(cells) >= 2:
                if any(k in cells[0] for k in ['정정', '최초']) and any(k in cells[0] for k in ['제출일', '공시일', '일자']):
                    date_match = re.search(r'(\d{4})[-/.](\d{2})[-/.](\d{2})', cells[1])
                    if date_match:
                        return ''.join(date_match.groups())
    except Exception:
        pass
    return None


def build_periodic_lineage(manifest_df: pd.DataFrame) -> dict:
    periodic_df = manifest_df[manifest_df['doc_group'] == 'periodic'].copy()
    lineage_map = {}

    groups = periodic_df.groupby(['corp_code', 'base_year', 'base_month', 'doc_subtype'])

    for (corp_code, base_year, base_month, doc_subtype), grp in groups:
        sorted_docs = grp.sort_values(by=['rcept_dt', 'rcept_no'], ascending=[True, True])
        docs_list = sorted_docs.to_dict('records')
        total_ver = len(docs_list)
        root_rcept_no = str(docs_list[0]['rcept_no'])
        lineage_id = f'lineage_periodic_{corp_code}_{int(base_year)}_{int(base_month)}_{doc_subtype}'

        history_summary = []
        for v_idx, doc in enumerate(docs_list, start=1):
            r_no = str(doc['rcept_no'])
            is_latest = (v_idx == total_ver)
            history_summary.append({
                'version': v_idx,
                'rcept_no': r_no,
                'rcept_dt': str(doc['rcept_dt']),
                'is_correction': bool(doc.get('is_correction', False)),
                'report_nm': str(doc.get('report_nm', '')),
                'is_latest': is_latest
            })

        for v_idx, doc in enumerate(docs_list, start=1):
            r_no = str(doc['rcept_no'])
            lineage_map[r_no] = {
                'doc_id': str(doc['doc_id']),
                'rcept_no': r_no,
                'corp_code': str(doc['corp_code']),
                'corp_name': str(doc['corp_name']),
                'doc_group': 'periodic',
                'doc_subtype': str(doc_subtype),
                'lineage_id': lineage_id,
                'root_rcept_no': root_rcept_no,
                'version_order': v_idx,
                'total_versions': total_ver,
                'is_latest': (v_idx == total_ver),
                'is_correction': bool(doc.get('is_correction', False)),
                'history': history_summary
            }

    return lineage_map


def build_event_lineage(manifest_df: pd.DataFrame, corpus_root: Path) -> dict:
    event_df = manifest_df[manifest_df['doc_group'] != 'periodic'].copy()
    event_df['clean_title'] = event_df['report_nm'].apply(clean_report_title)

    lineage_map = {}
    grouped = event_df.groupby(['corp_code', 'doc_group'])

    for (corp_code, doc_group), grp in grouped:
        docs = grp.sort_values(by=['rcept_dt', 'rcept_no'], ascending=[True, True]).to_dict('records')
        chains = []

        for doc in docs:
            r_no = str(doc['rcept_no'])
            is_corr = bool(doc.get('is_correction', False))
            c_title = doc['clean_title']
            flr_nm = str(doc.get('flr_nm', '')).strip()

            matched_chain_idx = None

            if is_corr:
                orig_dt = extract_orig_rcept_dt_from_xml(corpus_root, str(doc.get('file_path', '')))
                
                if orig_dt:
                    for c_idx, ch in enumerate(chains):
                        root_doc = ch[0]
                        if doc_group == 'holding':
                            if flr_nm and str(root_doc.get('flr_nm', '')).strip() != flr_nm:
                                continue
                        if str(root_doc['rcept_dt']) == orig_dt and root_doc['clean_title'] == c_title:
                            matched_chain_idx = c_idx
                            break

                if matched_chain_idx is None:
                    candidates = []
                    for c_idx, ch in enumerate(chains):
                        last_doc = ch[-1]
                        if doc_group == 'holding':
                            if flr_nm and str(last_doc.get('flr_nm', '')).strip() != flr_nm:
                                continue
                        if last_doc['clean_title'] == c_title:
                            candidates.append(c_idx)
                    if candidates:
                        matched_chain_idx = candidates[-1]

            if matched_chain_idx is not None:
                chains[matched_chain_idx].append(doc)
            else:
                chains.append([doc])

        for ch_idx, chain in enumerate(chains, start=1):
            total_ver = len(chain)
            root_rcept_no = str(chain[0]['rcept_no'])
            lineage_id = f'lineage_{doc_group}_{corp_code}_{root_rcept_no}'

            history_summary = []
            for v_idx, doc in enumerate(chain, start=1):
                r_no = str(doc['rcept_no'])
                is_latest = (v_idx == total_ver)
                history_summary.append({
                    'version': v_idx,
                    'rcept_no': r_no,
                    'rcept_dt': str(doc['rcept_dt']),
                    'is_correction': bool(doc.get('is_correction', False)),
                    'report_nm': str(doc.get('report_nm', '')),
                    'is_latest': is_latest
                })

            for v_idx, doc in enumerate(chain, start=1):
                r_no = str(doc['rcept_no'])
                lineage_map[r_no] = {
                    'doc_id': str(doc['doc_id']),
                    'rcept_no': r_no,
                    'corp_code': str(doc['corp_code']),
                    'corp_name': str(doc['corp_name']),
                    'doc_group': str(doc['doc_group']),
                    'doc_subtype': str(doc.get('doc_subtype', '')),
                    'lineage_id': lineage_id,
                    'root_rcept_no': root_rcept_no,
                    'version_order': v_idx,
                    'total_versions': total_ver,
                    'is_latest': (v_idx == total_ver),
                    'is_correction': bool(doc.get('is_correction', False)),
                    'history': history_summary
                }

    return lineage_map


def generate_full_lineage(manifest_path: str | Path, corpus_root: str | Path, out_path: str | Path) -> dict:
    manifest_path = Path(manifest_path)
    corpus_root = Path(corpus_root)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f'[1/3] manifest 로드: {manifest_path}')
    manifest_df = pd.read_json(
        manifest_path,
        lines=True,
        dtype={'corp_code': str, 'stock_code': str, 'rcept_no': str, 'rcept_dt': str}
    )

    print('[2/3] 정기공시 및 이벤트성 공시 통합 계보 테이블 생성 중...')
    periodic_lineage = build_periodic_lineage(manifest_df)
    event_lineage = build_event_lineage(manifest_df, corpus_root)

    full_lineage = {**periodic_lineage, **event_lineage}

    n_total = len(full_lineage)
    n_multi = sum(1 for v in full_lineage.values() if v['total_versions'] > 1)
    n_old = sum(1 for v in full_lineage.values() if not v['is_latest'])

    print(f'[3/3] 결과 저장: {out_path}')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(full_lineage, f, ensure_ascii=False, indent=2)

    print()
    print('===== 통합 문서 계보(Lineage) 테이블 구축 완료 =====')
    print(f'전체 공시 문서 수    : {n_total:,} 건')
    print(f'정정/다중 버전 문서  : {n_multi:,} 건')
    print(f'구버전(정정 전) 문서 : {n_old:,} 건 (최신본 우선 검색 대상)')
    print(f'산출물 파일          : {out_path.resolve()}')

    return full_lineage


def main():
    ap = argparse.ArgumentParser(description='DART 통합 문서 계보(Lineage) 테이블 생성기')
    ap.add_argument('--manifest', default='corpus/manifest.jsonl', help='manifest.jsonl 경로')
    ap.add_argument('--corpus-root', default='corpus', help='corpus 루트 경로')
    ap.add_argument('--out', default='doc_lineage.json', help='계보 결과 저장 경로 (.json)')
    args = ap.parse_args()

    generate_full_lineage(args.manifest, args.corpus_root, args.out)


if __name__ == '__main__':
    main()
