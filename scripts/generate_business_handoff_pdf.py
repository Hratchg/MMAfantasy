"""Generate PDF from BUSINESS_HANDOFF.md via pandoc → HTML → Chrome headless → PDF.

Audience: business stakeholders. Lighter, cleaner styling than the technical handoff —
larger body text, more whitespace, less code-block emphasis, table-friendly.
"""

import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
MD_PATH = os.path.join(REPO_ROOT, "BUSINESS_HANDOFF.md")
HTML_PATH = os.path.join(REPO_ROOT, "docs", "pdfs", "BUSINESS_HANDOFF.html")
PDF_PATH = os.path.join(REPO_ROOT, "docs", "pdfs", "BUSINESS_HANDOFF.pdf")

CSS = """
@page { size: Letter; margin: 0.85in 0.75in; }
body {
  font-family: Georgia, "Times New Roman", serif;
  font-size: 11.5px;
  line-height: 1.55;
  color: #1f2228;
  max-width: 7.5in;
  margin: 0 auto;
}
h1 {
  font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 24px;
  color: #0d1117;
  border-bottom: 3px solid #c0392b;
  padding-bottom: 10px;
  margin-top: 0;
  margin-bottom: 14px;
}
h2 {
  font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 16px;
  color: #2c3e50;
  margin-top: 28px;
  margin-bottom: 8px;
  page-break-after: avoid;
}
h3 {
  font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 13px;
  color: #34495e;
  margin-top: 18px;
  page-break-after: avoid;
}
p { margin: 8px 0; }
table {
  width: 100%;
  border-collapse: collapse;
  margin: 14px 0;
  font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 10.5px;
}
th {
  background-color: #2c3e50;
  color: white;
  padding: 8px 11px;
  text-align: left;
  font-weight: 600;
}
td { padding: 7px 11px; border-bottom: 1px solid #d6d9dc; vertical-align: top; }
tr:nth-child(even) td { background-color: #f7f8fa; }
code {
  background-color: #f4f4f4;
  padding: 1px 5px;
  font-family: "SF Mono", Menlo, Consolas, monospace;
  font-size: 10.5px;
  border-radius: 3px;
}
strong { color: #0d1117; }
hr { border: none; border-top: 1px solid #e1e4e8; margin: 24px 0; }
a { color: #2980b9; text-decoration: none; }
ul, ol { margin: 8px 0; padding-left: 24px; }
li { margin: 5px 0; }
blockquote {
  border-left: 3px solid #3498db;
  padding: 8px 14px;
  color: #555;
  margin: 12px 0;
  background-color: #f8fbfd;
  font-style: italic;
}
"""

CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    "/opt/homebrew/bin/chromium",
]


def find_chrome():
    for path in CHROME_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


def generate():
    chrome = find_chrome()
    if not chrome:
        print("ERROR: no Chrome / Chromium binary found in standard locations", file=sys.stderr)
        return False

    pandoc_html = subprocess.run(
        [
            "pandoc",
            MD_PATH,
            "--to=html5",
            "--standalone",
            "--metadata",
            "title=UFC Fight Prediction — Business Handoff",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    styled = pandoc_html.replace(
        "</head>",
        f"<style>{CSS}</style></head>",
        1,
    )

    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(styled)

    subprocess.run(
        [
            chrome,
            "--headless",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={PDF_PATH}",
            f"file://{HTML_PATH}",
        ],
        check=True,
        capture_output=True,
    )

    size_kb = os.path.getsize(PDF_PATH) / 1024
    print(f"PDF generated: {PDF_PATH} ({size_kb:.0f} KB)")
    return True


if __name__ == "__main__":
    sys.exit(0 if generate() else 1)
