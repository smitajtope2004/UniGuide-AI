from pypdf import PdfReader

pdf_path = "data/UniGuide_Sample_Academic_Rules.pdf"

reader = PdfReader(pdf_path)

text = ""

for page in reader.pages:
    text += page.extract_text()

print(text)