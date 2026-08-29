import os
from pypdf import PdfReader
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / '.env')

RAG_DIR = Path(os.getenv('RAG_DIR', Path(__file__).resolve().parents[1]))

ROOT = os.getenv('ARCHIVE_ROOT')
if not ROOT:
    raise SystemExit('ARCHIVE_ROOT is not set. Copy .env.example to .env and fill it in.')

text_pdf, scan_pdf, broken = [], [], []

for dirpath, _, files in os.walk(ROOT):
    for f in files:
        if not f.lower().endswith(".pdf"):
            continue
        path = os.path.join(dirpath, f)
        try:
            reader = PdfReader(path)
            pages = min(3, len(reader.pages))
            chars = sum(len(reader.pages[i].extract_text() or "") for i in range(pages))
            (text_pdf if chars > 200 else scan_pdf).append(path)
        except Exception as e:
            broken.append((path, str(e)[:60]))

print(f"С текстовым слоем: {len(text_pdf)}")
print(f"Похоже на сканы:   {len(scan_pdf)}")
print(f"Не открылись:      {len(broken)}")

with open(RAG_DIR / "pdf_text.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(text_pdf))
with open(RAG_DIR / "pdf_scan.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(scan_pdf))