"""Generate client-facing PDF from METHODOLOGY_CLIENT.md."""

import os
import markdown
from xhtml2pdf import pisa

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
MD_PATH = os.path.join(REPO_ROOT, "METHODOLOGY_CLIENT.md")
PDF_PATH = os.path.join(REPO_ROOT, "docs", "pdfs", "UFC_Elo_Methodology_Client.pdf")

CSS = """
@page {
    size: letter;
    margin: 0.9in 1in 0.9in 1in;
}

body {
    font-family: Georgia, 'Times New Roman', serif;
    font-size: 11px;
    line-height: 1.7;
    color: #1a1a1a;
}

h1 {
    font-family: Helvetica, Arial, sans-serif;
    font-size: 26px;
    color: #0d1b2a;
    border-bottom: 3px solid #c0392b;
    padding-bottom: 10px;
    margin-top: 36px;
    margin-bottom: 6px;
}

h2 {
    font-family: Helvetica, Arial, sans-serif;
    font-size: 18px;
    color: #0d1b2a;
    border-bottom: 1px solid #c0392b;
    padding-bottom: 4px;
    margin-top: 28px;
    margin-bottom: 8px;
}

h3 {
    font-family: Helvetica, Arial, sans-serif;
    font-size: 14px;
    color: #1a3a5c;
    margin-top: 20px;
    margin-bottom: 6px;
}

h4 {
    font-family: Helvetica, Arial, sans-serif;
    font-size: 12px;
    color: #1a3a5c;
    margin-top: 14px;
    margin-bottom: 4px;
    font-style: italic;
}

p {
    margin: 0 0 10px 0;
}

/* Tables */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 14px 0 18px 0;
    font-family: Helvetica, Arial, sans-serif;
}

th {
    background-color: #0d1b2a;
    color: #ffffff;
    padding: 8px 12px;
    text-align: left;
    font-size: 10px;
    font-weight: bold;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

td {
    padding: 7px 12px;
    border-bottom: 1px solid #e0e0e0;
    font-size: 10px;
    vertical-align: top;
}

tr:nth-child(even) td {
    background-color: #f5f7fa;
}

tr:first-child td {
    background-color: #eaf0fb;
    font-weight: bold;
}

/* Code and pre */
code {
    font-family: 'Courier New', Courier, monospace;
    background-color: #f0f4f8;
    padding: 2px 6px;
    font-size: 10px;
    border-radius: 2px;
    color: #c0392b;
}

pre {
    background-color: #f0f4f8;
    border-left: 4px solid #1a3a5c;
    padding: 12px 16px;
    font-family: 'Courier New', Courier, monospace;
    font-size: 9.5px;
    line-height: 1.6;
    margin: 12px 0;
    white-space: pre-wrap;
    word-wrap: break-word;
}

pre code {
    background: none;
    padding: 0;
    color: #1a1a1a;
}

/* Blockquote — used for call-out boxes */
blockquote {
    background-color: #fdf3e7;
    border-left: 4px solid #e67e22;
    padding: 10px 14px;
    margin: 14px 0;
    font-style: normal;
    font-size: 10.5px;
    color: #333;
}

blockquote p {
    margin: 0;
}

/* Lists */
ul, ol {
    margin: 6px 0 10px 20px;
    padding: 0;
}

li {
    margin-bottom: 4px;
    font-size: 11px;
}

/* Horizontal rule */
hr {
    border: none;
    border-top: 2px solid #e0e0e0;
    margin: 24px 0;
}

/* Strong emphasis */
strong {
    color: #0d1b2a;
    font-weight: bold;
}

em {
    color: #555;
}

/* Links */
a {
    color: #1a3a5c;
}

/* Note at top */
.note {
    background-color: #eaf4fb;
    border: 1px solid #aed6f1;
    padding: 10px 14px;
    margin-bottom: 20px;
    font-size: 10.5px;
}
"""


def generate():
    with open(MD_PATH, encoding="utf-8") as f:
        md_text = f.read()

    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "smarty", "toc"],
    )

    full_html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>{CSS}</style>
</head>
<body>
{html_body}
</body>
</html>"""

    with open(PDF_PATH, "wb") as out:
        result = pisa.CreatePDF(full_html, dest=out)

    if result.err:
        print(f"ERROR: {result.err} errors during PDF generation")
        return False

    size_kb = os.path.getsize(PDF_PATH) / 1024
    print(f"PDF generated: {PDF_PATH} ({size_kb:.0f} KB)")
    return True


if __name__ == "__main__":
    generate()
