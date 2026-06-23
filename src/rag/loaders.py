from pathlib import Path
from bs4 import BeautifulSoup
from pypdf import PdfReader

def load_text(path: Path) -> str:
    return path.read_text(encoding='utf-8')

def load_html(path: Path) -> str:
    soup = BeautifulSoup(path.read_text(encoding='utf-8'), 'html.parser')
    return soup.get_text('\n', strip=True)

def load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return '\n'.join(page.extract_text() or '' for page in reader.pages)

def load_document(path: Path) -> dict:
    suffix = path.suffix.lower()
    if suffix in {'.txt', '.md'}:
        text = load_text(path)
    elif suffix in {'.html', '.htm'}:
        text = load_html(path)
    elif suffix == '.pdf':
        text = load_pdf(path)
    else:
        raise ValueError(f'Unsupported format: {suffix}')
    return {'source_file': path.name, 'text': text}
