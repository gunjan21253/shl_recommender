"""
Diagnostic script: For each evaluation query, check whether the missed
assessments are in the retrieval pool (top-80 hybrid search candidates)
or completely outside it.

This tells us if the problem is RETRIEVAL (not found) vs RANKING (found but not selected).

Usage:
    python -m evaluation.diagnose --data "Gen_AI Dataset.xlsx"
"""
import argparse
import json
import logging
import re
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def normalize_url(url):
    if not url:
        return ""
    url = url.strip().rstrip("/").lower()
    m = re.search(r"/view/([^/?#]+)", url)
    return m.group(1) if m else url


def load_ground_truth(path, sheet=None):
    import pandas as pd
    p = Path(path)
    if p.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(p, sheet_name=sheet or 0, engine="openpyxl")
    else:
        df = pd.read_csv(p)
    df.columns = [str(c).strip() for c in df.columns]
    qcol = next((c for c in df.columns if "query" in c.lower()), df.columns[0])
    ucol = next((c for c in df.columns if "url" in c.lower() or "assessment" in c.lower()), df.columns[1])
    gt = defaultdict(list)
    for _, row in df.iterrows():
        q = str(row[qcol]).strip()
        u = str(row[ucol]).strip()
        if q and u and u != "nan":
            gt[q].append(u)
    return dict(gt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--sheet", default=None)
    args = parser.parse_args()

    gt = load_ground_truth(args.data, args.sheet)

    from recommender.engine import (
        _load_resources, hybrid_search,
        parse_constraints, apply_constraints, inject_must_consider, _assessments
    )
    _load_resources()

    LLM_WINDOW = 40  # how many candidates the LLM sees

    total_retrieval_miss = 0
    total_ranking_miss = 0
    total_relevant = 0
    total_hits = 0

    for qi, (query, relevant_urls) in enumerate(gt.items(), 1):
        relevant_slugs = {normalize_url(u) for u in relevant_urls}
        total_relevant += len(relevant_slugs)

        candidates = hybrid_search(query, top_k=80)
        candidates = inject_must_consider(candidates)

        constraints = parse_constraints(query)
        candidates_filtered = apply_constraints(candidates, constraints)

        candidate_slugs_80 = {normalize_url(c.get("url", "")) for c in candidates}
        candidate_slugs_filtered = {normalize_url(c.get("url", "")) for c in candidates_filtered}
        top20_slugs = {normalize_url(c.get("url", "")) for c in candidates[:LLM_WINDOW]}

        hits_in_top20 = relevant_slugs & top20_slugs
        hits_in_pool = relevant_slugs & candidate_slugs_80
        hits_in_filtered = relevant_slugs & candidate_slugs_filtered
        retrieval_misses = relevant_slugs - candidate_slugs_80
        ranking_misses = (relevant_slugs & candidate_slugs_80) - top20_slugs

        log.info(f"\n{'='*70}")
        log.info(f"[Q{qi}] {query[:80]}...")
        log.info(f"  Relevant: {len(relevant_slugs)} | In pool: {len(hits_in_pool)} | "
                 f"In top{LLM_WINDOW} (LLM sees): {len(hits_in_top20)} | "
                 f"Retrieval miss: {len(retrieval_misses)}")

        if retrieval_misses:
            log.info(f"  RETRIEVAL MISSES (not in top-80 at all):")
            for s in sorted(retrieval_misses):
                # Find its actual rank if we search deeper
                log.info(f"    - {s}")
            total_retrieval_miss += len(retrieval_misses)

        if ranking_misses:
            log.info(f"  RANKING MISSES (in pool but NOT in top-{LLM_WINDOW} sent to LLM):")
            for s in sorted(ranking_misses):
                idx = None
                for i, c in enumerate(candidates):
                    if normalize_url(c.get("url", "")) == s:
                        idx = i + 1
                        break
                log.info(f"    - {s} (rank #{idx} in retrieval)")
            total_ranking_miss += len(ranking_misses)

        in_top20_list = sorted(hits_in_top20)
        if in_top20_list:
            log.info(f"  IN TOP-{LLM_WINDOW} (LLM should pick these):")
            for s in in_top20_list:
                idx = None
                for i, c in enumerate(candidates):
                    if normalize_url(c.get("url", "")) == s:
                        idx = i + 1
                        break
                log.info(f"    + {s} (rank #{idx})")

    log.info(f"\n{'='*70}")
    log.info(f"SUMMARY")
    log.info(f"  Total relevant assessments: {total_relevant}")
    log.info(f"  RETRIEVAL MISSES (not in pool at all): {total_retrieval_miss}")
    log.info(f"  RANKING MISSES (in pool but not top-{LLM_WINDOW}): {total_ranking_miss}")
    log.info(f"  => Retrieval is the bottleneck" if total_retrieval_miss > total_ranking_miss
             else f"  => LLM ranking/top-{LLM_WINDOW} cutoff is the bottleneck")


if __name__ == "__main__":
    main()
