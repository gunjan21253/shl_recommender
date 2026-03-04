"""
Evaluation Script
=================
Computes Mean Recall@10 against the labeled training set.
Supports both CSV and Excel (.xlsx) training files.

Usage:
    python -m evaluation.evaluate --data data/train.csv --direct
    python -m evaluation.evaluate --data Gen_AI Dataset.xlsx --direct
    python -m evaluation.evaluate --data Gen_AI Dataset.xlsx --sheet "Train-Set" --direct
    # Run on 10 random queries from the train set:
    python -m evaluation.evaluate --data "Gen_AI Dataset.xlsx" --sheet "Train-Set" --max-queries 10 --random --direct
"""

import argparse
import json
import logging
import math
import random
import re
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd
import requests

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────
# URL Normalization (critical for correct eval)
# ─────────────────────────────────────────────

def normalize_url(url: str) -> str:
    """
    Normalize SHL URLs by extracting the slug (last path component).

    Training data has mixed URL formats:
      /products/product-catalog/view/java-8-new/
      /solutions/products/product-catalog/view/java-8-new
    """
    if not url:
        return ""
    url = url.strip().rstrip("/").lower()
    match = re.search(r"/view/([^/?#]+)", url)
    if match:
        return match.group(1)
    return url


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

def recall_at_k(predicted_urls: list, relevant_urls: set, k: int = 10) -> float:
    """Recall@K = |predicted_top_k intersection relevant| / |relevant|"""
    if not relevant_urls:
        return 0.0
    predicted_slugs = {normalize_url(u) for u in predicted_urls[:k]}
    relevant_slugs = {normalize_url(u) for u in relevant_urls}
    hits = len(predicted_slugs & relevant_slugs)
    return hits / len(relevant_slugs)


def precision_at_k(predicted_urls: list, relevant_urls: set, k: int = 10) -> float:
    if not predicted_urls:
        return 0.0
    predicted_slugs = {normalize_url(u) for u in predicted_urls[:k]}
    relevant_slugs = {normalize_url(u) for u in relevant_urls}
    hits = len(predicted_slugs & relevant_slugs)
    return hits / min(k, len(predicted_urls))


def ndcg_at_k(predicted_urls: list, relevant_urls: set, k: int = 10) -> float:
    """Normalized Discounted Cumulative Gain."""
    relevant_slugs = {normalize_url(u) for u in relevant_urls}

    dcg = 0.0
    for i, url in enumerate(predicted_urls[:k]):
        slug = normalize_url(url)
        if slug in relevant_slugs:
            dcg += 1.0 / math.log2(i + 2)

    ideal_hits = min(k, len(relevant_slugs))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0


# ─────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────

def _load_dataframe(path: str, sheet: str = None) -> pd.DataFrame:
    """Load CSV or Excel into a DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Training data not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in (".xlsx", ".xls"):
        if sheet is not None:
            df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
        else:
            # Default: first sheet (usually the train/labeled set)
            df = pd.read_excel(path, sheet_name=0, engine="openpyxl")
    else:
        raise ValueError(f"Unsupported format: {suffix}. Use .csv or .xlsx")

    return df


def load_ground_truth(data_path: str, sheet: str = None) -> dict:
    """Load labeled training data. Returns: {query_string: [list_of_relevant_urls]}"""
    df = _load_dataframe(data_path, sheet=sheet)

    df.columns = [str(c).strip() for c in df.columns]
    query_col = next((c for c in df.columns if "query" in c.lower()), df.columns[0])
    url_col = next(
        (c for c in df.columns if "url" in c.lower() or "assessment" in c.lower()),
        df.columns[1] if len(df.columns) > 1 else df.columns[0]
    )

    ground_truth = defaultdict(list)
    for _, row in df.iterrows():
        query = str(row[query_col]).strip()
        url = str(row[url_col]).strip()
        if query and url and url != "nan":
            ground_truth[query].append(url)

    log.info(f"Loaded {len(ground_truth)} queries with ground truth from {data_path}")
    for q, urls in ground_truth.items():
        log.info(f"  '{q[:60]}...' -> {len(urls)} relevant assessments")

    return dict(ground_truth)


# ─────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────

def predict_via_api(query: str, api_url: str) -> list:
    try:
        response = requests.post(
            f"{api_url}/recommend",
            json={"query": query},
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        assessments = data.get("recommended_assessments", [])
        return [a["url"] for a in assessments]
    except Exception as e:
        log.error(f"API call failed for query '{query[:50]}...': {e}")
        return []


def predict_direct(query: str) -> list:
    from recommender.engine import get_recommendations
    results = get_recommendations(query, top_k=10)
    return [r["url"] for r in results]


# ─────────────────────────────────────────────
# Main Evaluation Loop
# ─────────────────────────────────────────────

def evaluate(
    data_path: str,
    api_url: str = None,
    direct: bool = False,
    sheet: str = None,
    k: int = 10,
    verbose: bool = True,
    max_queries: int = None,
    random_sample: bool = False,
) -> dict:
    ground_truth = load_ground_truth(data_path, sheet=sheet)

    if max_queries is not None:
        queries_list = list(ground_truth.items())
        if random_sample:
            random.shuffle(queries_list)
        n = min(max_queries, len(queries_list))
        chosen = dict(queries_list[:n])
        log.info("Using %d queries (max_queries=%s, random=%s)", len(chosen), max_queries, random_sample)
        ground_truth = chosen

    all_recalls = []
    all_precisions = []
    all_ndcgs = []
    per_query_results = []

    for i, (query, relevant_urls) in enumerate(ground_truth.items()):
        log.info(f"\n[{i+1}/{len(ground_truth)}] Query: {query[:70]}...")

        if direct:
            predicted_urls = predict_direct(query)
        elif api_url:
            predicted_urls = predict_via_api(query, api_url)
        else:
            raise ValueError("Provide either --api URL or --direct flag")

        rec = recall_at_k(predicted_urls, set(relevant_urls), k)
        prec = precision_at_k(predicted_urls, set(relevant_urls), k)
        ndcg = ndcg_at_k(predicted_urls, set(relevant_urls), k)

        all_recalls.append(rec)
        all_precisions.append(prec)
        all_ndcgs.append(ndcg)

        if verbose:
            relevant_slugs = {normalize_url(u) for u in relevant_urls}
            predicted_slugs = [normalize_url(u) for u in predicted_urls]
            hits = [s for s in predicted_slugs if s in relevant_slugs]
            misses = [s for s in relevant_slugs if s not in set(predicted_slugs)]

            log.info(f"  Relevant: {len(relevant_urls)} | Predicted: {len(predicted_urls)}")
            log.info(f"  Recall@{k}: {rec:.3f} | Precision@{k}: {prec:.3f} | NDCG@{k}: {ndcg:.3f}")
            if hits:
                log.info(f"  Hits: {hits}")
            if misses:
                log.info(f"  Missed: {misses}")

        per_query_results.append({
            "query": query,
            "recall": rec,
            "precision": prec,
            "ndcg": ndcg,
            "n_relevant": len(relevant_urls),
            "n_predicted": len(predicted_urls),
            "predicted_urls": predicted_urls,
        })

    mean_recall = sum(all_recalls) / len(all_recalls) if all_recalls else 0
    mean_precision = sum(all_precisions) / len(all_precisions) if all_precisions else 0
    mean_ndcg = sum(all_ndcgs) / len(all_ndcgs) if all_ndcgs else 0

    print("\n" + "=" * 60)
    print(f"  EVALUATION RESULTS (k={k})")
    print("=" * 60)
    print(f"  Mean Recall@{k}:    {mean_recall:.4f}  <- Primary metric")
    print(f"  Mean Precision@{k}: {mean_precision:.4f}")
    print(f"  Mean NDCG@{k}:      {mean_ndcg:.4f}")
    print(f"  Queries evaluated:  {len(all_recalls)}")
    print("=" * 60)

    print("\nPer-query Recall:")
    for r in per_query_results:
        print(f"  [{r['recall']:.2f}] {r['query'][:65]}...")

    return {
        "mean_recall": mean_recall,
        "mean_precision": mean_precision,
        "mean_ndcg": mean_ndcg,
        "per_query": per_query_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate SHL Recommender")
    parser.add_argument(
        "--data",
        required=True,
        help="Path to labeled training data: CSV or Excel (.xlsx), e.g. Gen_AI Dataset.xlsx or data/train.csv",
    )
    parser.add_argument(
        "--sheet",
        default=None,
        help="Excel sheet name (e.g. 'Train-Set' for Gen_AI Dataset.xlsx). If not set, first sheet is used.",
    )
    parser.add_argument("--api", default=None, help="API base URL")
    parser.add_argument("--direct", action="store_true", help="Use engine directly")
    parser.add_argument("--k", type=int, default=10, help="K for Recall@K (default: 10)")
    parser.add_argument("--max-queries", type=int, default=None, help="Use at most N queries (e.g. 10). With --random, sample N at random from train set.")
    parser.add_argument("--random", action="store_true", help="When using --max-queries, sample randomly instead of first N.")
    parser.add_argument("--output", default=None, help="Save results to JSON file")
    args = parser.parse_args()

    results = evaluate(
        data_path=args.data,
        api_url=args.api,
        direct=args.direct,
        sheet=args.sheet,
        k=args.k,
        max_queries=args.max_queries,
        random_sample=args.random,
    )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        log.info(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
