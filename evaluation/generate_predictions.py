"""
Generate Test Set Predictions (SHL assignment submission)
=========================================================
Per assignment: use the UNLABELED TEST SET (9 queries) from the same dataset
as the train data. Produce one CSV in the exact format required (Appendix 3).

Gen_AI Dataset.xlsx contains:
  - Train-Set sheet: 10 labeled queries (for evaluation/iteration).
  - Test-Set sheet:  9 unlabeled queries (for submission).

Output format:
    Query,Assessment_url
    "<query text 1>",https://www.shl.com/...
    "<query text 1>",https://www.shl.com/...
    "<query text 2>",https://www.shl.com/...

- Min 1, max 10 recommendations per query.
- Column names must be exactly: Query, Assessment_url

Usage:
    # From Gen_AI Dataset.xlsx using the Test-Set sheet (9 queries)
    python -m evaluation.generate_predictions --test "Gen_AI Dataset.xlsx" --sheet "Test-Set" --output submission.csv --direct

    # CSV with one column of test queries
    python -m evaluation.generate_predictions --test data/test_queries.csv --output submission.csv --direct

    # Via API
    python -m evaluation.generate_predictions --test "Gen_AI Dataset.xlsx" --sheet "Test-Set" --output submission.csv --api https://shl-recommender-tkj0.onrender.com
"""

import argparse
import csv
import logging
import time
import sys
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

sys.path.insert(0, str(Path(__file__).parent.parent))

# Submission format (Appendix 3)
QUERY_COL = "Query"
ASSESSMENT_URL_COL = "Assessment_url"


def load_test_queries(path: str, sheet: str = None) -> list:
    """Load test query strings from CSV or Excel. For Excel, use sheet (e.g. 'Test-Set') for unlabeled test set."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Test data not found: {path}")

    if p.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path, sheet_name=sheet or 0, engine="openpyxl")
    else:
        df = pd.read_csv(path)

    df.columns = [str(c).strip() for c in df.columns]
    query_col = next((c for c in df.columns if "query" in c.lower()), df.columns[0])
    queries = df[query_col].astype(str).dropna().replace("nan", "").str.strip()
    queries = queries[queries != ""].unique().tolist()
    log.info(f"Loaded {len(queries)} test queries from {path}" + (f" (sheet={sheet})" if sheet else ""))
    return queries


def get_predictions_via_api(query: str, api_url: str) -> list:
    try:
        response = requests.post(
            f"{api_url}/recommend",
            json={"query": query},
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        return [a["url"] for a in data.get("recommended_assessments", [])]
    except Exception as e:
        log.error(f"API error for '{query[:50]}...': {e}")
        return []


def get_predictions_direct(query: str) -> list:
    from recommender.engine import get_recommendations
    results = get_recommendations(query, top_k=10)
    return [r["url"] for r in results]


def generate_predictions(
    test_path: str,
    output_csv: str,
    sheet: str = None,
    api_url: str = None,
    direct: bool = False,
    delay: float = 1.0,
):
    queries = load_test_queries(test_path, sheet=sheet)

    if direct:
        from recommender.engine import get_llm_status
        llm_ok, llm_info = get_llm_status()
        if llm_ok:
            log.info("LLM re-ranking: ENABLED (model=%s) — test set will use Gemini for analysis + rerank", llm_info)
        else:
            log.warning("LLM re-ranking: DISABLED (%s) — test set will use retrieval order only (no Gemini)", llm_info)

    rows = []
    for i, query in enumerate(queries):
        log.info(f"[{i+1}/{len(queries)}] Processing: {query[:70]}...")

        if direct:
            urls = get_predictions_direct(query)
        elif api_url:
            urls = get_predictions_via_api(query, api_url)
            time.sleep(delay)
        else:
            raise ValueError("Provide --api or --direct")

        log.info(f"  Got {len(urls)} recommendations")
        for url in urls:
            rows.append({QUERY_COL: query, ASSESSMENT_URL_COL: url})

    # Write CSV in exact Appendix 3 format: "Query","Assessment_url" (no spaces in header)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow([QUERY_COL, ASSESSMENT_URL_COL])
        for r in rows:
            writer.writerow([r[QUERY_COL], r[ASSESSMENT_URL_COL]])

    df = pd.DataFrame(rows, columns=[QUERY_COL, ASSESSMENT_URL_COL])
    log.info(f"\nSaved {len(rows)} prediction rows to {output_csv}")
    log.info(f"  {len(queries)} queries x avg {len(rows)/max(len(queries),1):.1f} recommendations each")

    print("\nFirst 20 rows of predictions (submission format):")
    print(df.head(20).to_string(index=False))

    return pd.DataFrame(rows, columns=[QUERY_COL, ASSESSMENT_URL_COL])


def main():
    parser = argparse.ArgumentParser(description="Generate SHL test set predictions (submission CSV)")
    parser.add_argument("--test", required=True, help="Path to test data: CSV or Excel (e.g. Gen_AI Dataset.xlsx)")
    parser.add_argument("--sheet", default=None, help="Excel sheet name for test set (e.g. 'Test-Set'). If omitted, first sheet is used.")
    parser.add_argument("--output", default="submission.csv", help="Output CSV path (Appendix 3 format)")
    parser.add_argument("--api", default=None, help="API base URL")
    parser.add_argument("--direct", action="store_true", help="Use engine directly (no API)")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between API calls (s)")
    args = parser.parse_args()

    generate_predictions(
        test_path=args.test,
        output_csv=args.output,
        sheet=args.sheet,
        api_url=args.api,
        direct=args.direct,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
