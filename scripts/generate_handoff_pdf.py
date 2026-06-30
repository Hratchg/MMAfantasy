"""Generate PDF from TECHNICAL_HANDOFF.md via pandoc → HTML → Chrome headless → PDF.

Replaces the xhtml2pdf path because xhtml2pdf 0.2.16+ pulls in svglib → rlpycairo →
pycairo whose wheel doesn't exist on Python 3.13 / arm64 and the source build requires
pkg-config + cairo system headers. Pandoc + Chrome headless is already installed on
this machine and has no dependency-resolution problems.
"""

import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
MD_PATH = os.path.join(REPO_ROOT, "TECHNICAL_HANDOFF.md")
HTML_PATH = os.path.join(REPO_ROOT, "docs", "pdfs", "TECHNICAL_HANDOFF.html")
PDF_PATH = os.path.join(REPO_ROOT, "docs", "pdfs", "TECHNICAL_HANDOFF.pdf")

CSS = """
@page { size: Letter; margin: 0.6in 0.5in; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 10.5px;
  line-height: 1.5;
  color: #222;
  max-width: 7.5in;
  margin: 0 auto;
  padding: 0;
}
h1 { font-size: 22px; color: #1a1a2e; border-bottom: 2px solid #c0392b; padding-bottom: 8px; margin-top: 28px; }
h2 { font-size: 16px; color: #2c3e50; border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; margin-top: 24px; }
h3 { font-size: 13px; color: #34495e; margin-top: 18px; }
h4 { font-size: 12px; color: #34495e; margin-top: 14px; }
table { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 9.5px; }
th { background-color: #2c3e50; color: white; padding: 7px 9px; text-align: left; }
td { padding: 5px 9px; border-bottom: 1px solid #ddd; vertical-align: top; }
tr:nth-child(even) td { background-color: #f8f9fa; }
code {
  background-color: #f4f4f4;
  padding: 1px 5px;
  font-family: "SF Mono", Menlo, Consolas, Courier, monospace;
  font-size: 9.5px;
  border-radius: 3px;
}
pre {
  background-color: #f4f4f4;
  padding: 10px 12px;
  border-left: 3px solid #c0392b;
  font-family: "SF Mono", Menlo, Consolas, Courier, monospace;
  font-size: 9px;
  line-height: 1.4;
  overflow-x: auto;
}
pre code { background: none; padding: 0; }
blockquote {
  border-left: 3px solid #3498db;
  padding: 8px 14px;
  color: #555;
  margin: 12px 0;
  background-color: #f8fbfd;
}
strong { color: #1a1a2e; }
hr { border: none; border-top: 1px solid #ddd; margin: 20px 0; }
a { color: #2980b9; text-decoration: none; }
ul, ol { margin: 8px 0; padding-left: 24px; }
li { margin: 3px 0; }
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
            "title=UFC Fight Prediction — Technical Handoff",
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
