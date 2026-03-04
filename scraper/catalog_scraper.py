"""
SHL Catalog Scraper
===================
Scrapes all Individual Test Solutions from SHL's product catalog
using requests + BeautifulSoup (no browser needed).

The SHL catalog uses URL-based pagination:
  - type=1  -> Individual Test Solutions
  - type=2  -> Pre-packaged Job Solutions (ignored per assignment)
  - start=N -> offset (12 results per page)

Usage:
    python -m scraper.catalog_scraper
"""

import json
import time
import re
import logging
from pathlib import Path

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CATALOG_BASE = "https://www.shl.com/solutions/products/product-catalog/"
PAGE_SIZE = 12
OUTPUT_FILE = Path(__file__).parent.parent / "data" / "shl_catalog.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

TEST_TYPE_MAP = {
    "A": "Ability and Aptitude",
    "B": "Biodata and Situational Judgement",
    "C": "Competencies",
    "D": "Development and 360",
    "E": "Assessment Exercises",
    "K": "Knowledge and Skills",
    "P": "Personality and Behavior",
    "S": "Simulations",
}


def normalize_url(url: str) -> str:
    if not url:
        return url
    url = url.strip().rstrip("/")
    if url.startswith("http://"):
        url = url.replace("http://", "https://", 1)
    if not url.startswith("http"):
        url = f"https://www.shl.com{url}"
    return url


def parse_duration(text: str) -> int:
    if not text:
        return 0
    text = text.lower()
    hours = re.search(r"(\d+)\s*hour", text)
    mins = re.search(r"(\d+)\s*min", text)
    total = 0
    if hours:
        total += int(hours.group(1)) * 60
    if mins:
        total += int(mins.group(1))
    if total == 0:
        m = re.search(r"\d+", text)
        if m:
            total = int(m.group())
    return total


def scrape_detail_page(session: requests.Session, url: str) -> dict:
    """Fetch an assessment detail page for description and duration."""
    try:
        resp = session.get(url, headers=HEADERS, timeout=20)
        if resp.status_code != 200:
            log.debug(f"Detail page {resp.status_code}: {url}")
            return {"description": "", "duration": 0, "languages": []}

        soup = BeautifulSoup(resp.text, "html.parser")

        description = ""
        for selector in [
            "div.product-catalogue-training-catalogue__description",
            "div.product-hero__description",
            "[class*='description']",
            "main p",
        ]:
            el = soup.select_one(selector)
            if el:
                text = el.get_text(strip=True)
                if len(text) > 30:
                    description = text
                    break

        duration = 0
        body_text = soup.get_text()
        match = re.search(
            r"(?:completion|test|assessment|approximate)[\s\w]*?time[:\s]*?(\d+)\s*(?:to\s*\d+\s*)?min",
            body_text, re.IGNORECASE,
        )
        if match:
            duration = int(match.group(1))
        else:
            match = re.search(r"(\d+)\s*minutes", body_text, re.IGNORECASE)
            if match:
                val = int(match.group(1))
                if 1 <= val <= 300:
                    duration = val

        return {"description": description, "duration": duration}

    except Exception as e:
        log.debug(f"Detail page error {url}: {e}")
        return {"description": "", "duration": 0}


def parse_catalog_table(soup: BeautifulSoup) -> list:
    """Parse assessment rows from the catalog page HTML."""
    assessments = []

    tables = soup.select("table")
    for table in tables:
        header_row = table.select_one("thead tr, tr:first-child")
        if not header_row:
            continue

        header_text = header_row.get_text(strip=True).lower()
        is_individual = "individual" in header_text

        rows = table.select("tbody tr")
        if not rows:
            all_rows = table.select("tr")
            rows = all_rows[1:] if len(all_rows) > 1 else []

        for row in rows:
            cells = row.select("td")
            if not cells:
                continue

            link = row.select_one("a[href]")
            if not link:
                continue

            name = link.get_text(strip=True)
            href = link.get("href", "")
            if not href or "/view/" not in href:
                continue

            url = normalize_url(href)

            type_spans = row.select("span.product-catalogue__key")
            if not type_spans:
                type_spans = row.select("span[class*='key']")

            test_types = []
            for span in type_spans:
                letter = span.get_text(strip=True).upper()
                if letter in TEST_TYPE_MAP:
                    test_types.append(letter)

            if not test_types:
                for cell in cells:
                    cell_text = cell.get_text(strip=True)
                    for letter in cell_text:
                        if letter.upper() in TEST_TYPE_MAP and letter.upper() not in test_types:
                            if len(cell_text) <= 20:
                                test_types.append(letter.upper())

            remote_support = "No"
            adaptive_support = "No"

            if len(cells) >= 3:
                for ci, cell in enumerate(cells):
                    cell_html = str(cell)
                    has_check = (
                        "catalogue__circle--yes" in cell_html
                        or "circle--yes" in cell_html
                        or "check" in cell_html.lower()
                    )
                    cell_header = ""
                    if table.select_one("thead"):
                        th_cells = table.select("thead th")
                        if ci < len(th_cells):
                            cell_header = th_cells[ci].get_text(strip=True).lower()

                    if has_check:
                        if "remote" in cell_header:
                            remote_support = "Yes"
                        elif "adaptive" in cell_header or "irt" in cell_header:
                            adaptive_support = "Yes"
                        elif ci == len(cells) - 3:
                            remote_support = "Yes"
                        elif ci == len(cells) - 2:
                            adaptive_support = "Yes"

            assessments.append({
                "name": name,
                "url": url,
                "test_types": test_types,
                "test_types_expanded": [TEST_TYPE_MAP.get(t, t) for t in test_types],
                "remote_support": remote_support,
                "adaptive_support": adaptive_support,
                "description": "",
                "duration": 0,
                "is_individual": is_individual,
            })

    return assessments


def scrape_all_assessments(skip_details: bool = False, max_pages: int = 50) -> list:
    """
    Scrape both Individual Test Solutions (type=1) and Pre-packaged
    Job Solutions (type=2) from the SHL catalog.

    The training data includes Pre-packaged solutions as relevant,
    so both types are needed for accurate recommendations.
    """
    session = requests.Session()
    all_assessments = []
    seen_urls = set()

    catalog_types = [
        (1, "Individual Test Solutions"),
        (2, "Pre-packaged Job Solutions"),
    ]

    for cat_type, cat_name in catalog_types:
        log.info(f"\n=== Scraping {cat_name} (type={cat_type}) ===")

        for page in range(max_pages):
            start = page * PAGE_SIZE
            url = f"{CATALOG_BASE}?type={cat_type}&start={start}"

            log.info(f"--- Page {page + 1} (start={start}) ---")

            try:
                resp = session.get(url, headers=HEADERS, timeout=30)
                if resp.status_code != 200:
                    log.error(f"HTTP {resp.status_code} for {url}")
                    break
            except Exception as e:
                log.error(f"Request failed: {e}")
                break

            soup = BeautifulSoup(resp.text, "html.parser")
            page_assessments = parse_catalog_table(soup)

            if not page_assessments:
                log.info("No assessments found on page, done paginating")
                break

            new_count = 0
            for a in page_assessments:
                if a["url"] not in seen_urls:
                    seen_urls.add(a["url"])
                    all_assessments.append(a)
                    new_count += 1

            log.info(f"  Found {len(page_assessments)} rows, {new_count} new (total: {len(all_assessments)})")

            # For Pre-packaged (type=2), first page often duplicates type=1's first page;
            # keep paginating (start=12, 24, ...) to get the rest of the Pre-packaged list.
            if new_count == 0 and (cat_type != 2 or page > 2):
                log.info("No new assessments, done paginating")
                break

            time.sleep(0.5)

    log.info(f"\nTotal assessments scraped: {len(all_assessments)}")

    if not skip_details and all_assessments:
        log.info("Fetching detail pages for descriptions and durations...")
        for i, assessment in enumerate(all_assessments):
            if i % 25 == 0:
                log.info(f"  Detail pages: {i}/{len(all_assessments)}")
            details = scrape_detail_page(session, assessment["url"])
            assessment.update(details)
            time.sleep(0.3)

    for a in all_assessments:
        a.pop("is_individual", None)

    return all_assessments


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    log.info("Starting SHL catalog scrape (Individual + Pre-packaged)...")
    assessments = scrape_all_assessments(skip_details=False)

    if len(assessments) < 377:
        log.warning(
            f"Only scraped {len(assessments)} assessments - expected 440+. "
            "The SHL website structure may have changed."
        )
    else:
        log.info(f"Scraped {len(assessments)} assessments")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(assessments, f, indent=2, ensure_ascii=False)

    log.info(f"Saved to {OUTPUT_FILE}")

    type_counts = {}
    for a in assessments:
        for t in a.get("test_types", []):
            type_counts[t] = type_counts.get(t, 0) + 1

    log.info("Test type distribution:")
    for t, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        log.info(f"  {t} ({TEST_TYPE_MAP.get(t, '?')}): {count}")


if __name__ == "__main__":
    main()
