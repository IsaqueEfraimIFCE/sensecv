from pypdf import PdfReader
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
r = PdfReader(os.path.join(base_dir, "RAL18_Loquercio.pdf"))
print("PAGES:", len(r.pages))
out = []
for p in r.pages:
    out.append(p.extract_text())
text = "\n".join(out)
with open(os.path.join(base_dir, "paper_text.txt"), "w", encoding="utf-8") as f:
    f.write(text)
print("CHARS:", len(text))
print(text[:3000])
