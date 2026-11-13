import markdown2
import pdfplumber
from bs4 import BeautifulSoup
from docx import Document


def extract_md_text(path: str) -> str:
    with open(path, 'r', encoding = 'utf-8') as f:
        content = f.read()
    html = markdown2.markdown(content)
    soup = BeautifulSoup(html, 'html.parser')
    return soup.get_text()


def extract_pdf_text(path: str) -> str:
    text = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text().strip()
            if page_text:
                text.append(page_text)
    return '\n'.join(text)

def extract_word_text(path: str) -> list[str]:
    doc = Document(path)
    paragraphs = []
    for para in doc.paragraphs:
        para_text = para.text.strip()
        para_text = para_text.replace('\u3000', ' ')
        para_text = para_text.replace(r'\s+', ' ')

        if para_text:
            paragraphs.append(para_text)
    return paragraphs

def extract_file_text(path: str) -> str|list[str]:
# 根据文件类型决定提取文件的方法
    if path.endswith('.md'):
        return extract_md_text(path)
    elif path.endswith('.pdf'):
        return extract_pdf_text(path)
    elif path.endswith('.docx'):
        return extract_word_text(path)
    else:
        raise ValueError(f'暂不支持{path}的文件类型！')
