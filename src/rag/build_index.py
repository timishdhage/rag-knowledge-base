from .ingest import ingest_folder
from .vectorstore import VectorStore

def build(folder='docs'):
    chunks = ingest_folder(folder)
    for c in chunks:
        c['id'] = f"{c['source_file']}::{c['chunk_index']}"
    store = VectorStore()
    if chunks:
        store.add_chunks(chunks)
    return len(chunks)

if __name__ == '__main__':
    print(build())
