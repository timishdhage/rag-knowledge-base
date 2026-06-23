import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def embed_texts(texts):
    resp = client.embeddings.create(model='text-embedding-3-small', input=texts)
    return [item.embedding for item in resp.data]
