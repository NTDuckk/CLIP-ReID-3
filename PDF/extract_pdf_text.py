import fitz
import os

pdf_dir = os.path.dirname(os.path.abspath(__file__))

for filename in os.listdir(pdf_dir):
    if not filename.endswith('.pdf'):
        continue
    if 'PromptSG' in filename:
        print(f"Skipping: {filename}")
        continue

    pdf_path = os.path.join(pdf_dir, filename)
    doc = fitz.open(pdf_path)

    full_text = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        full_text.append(f"\n{'='*50}\nPage {page_num + 1}\n{'='*50}\n{text}")

    stem = os.path.splitext(filename)[0]
    output_path = os.path.join(pdf_dir, f"{stem}_extracted.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(full_text))

    print(f"Extracted {len(doc)} pages from '{filename}' -> '{os.path.basename(output_path)}'")
    print(f"  Total characters: {sum(len(t) for t in full_text)}")
