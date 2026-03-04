"""
Export SUBMISSION_APPROACH.md to HTML for printing to PDF.

Run from project root:
    python scripts/export_approach_to_html.py

Then open SUBMISSION_APPROACH.html in a browser and use Print → Save as PDF.
Keep to 2 pages when printing (adjust margins/scale if needed).
"""

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MD_FILE = PROJECT_ROOT / "SUBMISSION_APPROACH.md"
OUT_HTML = PROJECT_ROOT / "SUBMISSION_APPROACH.html"


def md_to_html(text: str) -> str:
    """Simple markdown to HTML for the approach doc (headings, bold, lists, code)."""
    html = []
    in_list = False
    for line in text.splitlines():
        line = line.rstrip()
        if not line:
            if in_list:
                html.append("</ul>")
                in_list = False
            html.append("<p></p>")
            continue
        if line.startswith("# "):
            if in_list:
                html.append("</ul>")
                in_list = False
            html.append(f'<h1>{line[2:].strip()}</h1>')
        elif line.startswith("## "):
            if in_list:
                html.append("</ul>")
                in_list = False
            html.append(f'<h2>{line[3:].strip()}</h2>')
        elif line.startswith("### "):
            if in_list:
                html.append("</ul>")
                in_list = False
            html.append(f'<h3>{line[4:].strip()}</h3>')
        elif line.strip().startswith("- ") or line.strip().startswith("* "):
            if not in_list:
                html.append("<ul>")
                in_list = True
            content = line.strip()[2:].strip()
            content = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", content)
            content = re.sub(r"`(.+?)`", r"<code>\1</code>", content)
            html.append(f"<li>{content}</li>")
        else:
            if in_list:
                html.append("</ul>")
                in_list = False
            content = line
            content = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", content)
            content = re.sub(r"`(.+?)`", r"<code>\1</code>", content)
            html.append(f"<p>{content}</p>")
    if in_list:
        html.append("</ul>")
    return "\n".join(html)


def main():
    if not MD_FILE.exists():
        print(f"Not found: {MD_FILE}")
        return 1
    text = MD_FILE.read_text(encoding="utf-8")
    body = md_to_html(text)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SHL Assessment Recommender — Approach</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 700px; margin: 1in auto; padding: 0 1em; font-size: 11pt; line-height: 1.4; }}
h1 {{ font-size: 14pt; margin-top: 0.5em; }}
h2 {{ font-size: 12pt; margin-top: 0.8em; }}
h3 {{ font-size: 11pt; }}
ul {{ margin: 0.3em 0; padding-left: 1.2em; }}
p {{ margin: 0.3em 0; }}
code {{ background: #f0f0f0; padding: 0.1em 0.3em; font-size: 0.95em; }}
@media print {{ body {{ margin: 0.5in; }} }}
</style>
</head>
<body>
{body}
</body>
</html>"""
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"Wrote {OUT_HTML}")
    print("Open it in a browser and use Print -> Save as PDF (keep to 2 pages).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
