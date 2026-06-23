from pathlib import Path
from .loaders import load_document
from .chunking import fixed_size_chunks

def ingest_folder(folder: str):
    docs = []
    for path in Path(folder).glob('*'):
        if path.is_file():
            try:
                doc = load_document(path)
                chunks = fixed_size_chunks(doc['text'], doc['source_file'])
                for c in chunks:
                    docs.append({'source_file': c.source_file, 'chunk_index': c.chunk_index, 'strategy': c.strategy, 'text': c.text})
            except Exception:
                pass
    return docs
