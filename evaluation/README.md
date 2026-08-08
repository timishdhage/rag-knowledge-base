# RAG Evaluation

This directory contains a small, version-controlled evaluation set for the document-grounded assistant.

## Dataset format

Each item contains:

- `id`: stable evaluation identifier.
- `question`: user question.
- `expected_source`: document or section that should be retrieved.
- `expected_answer_points`: facts an acceptable answer should cover.
- `should_refuse`: whether the system should decline because evidence is insufficient.

## Initial evaluation process

1. Ingest the evaluation documents.
2. Run each question through the retrieval and generation pipeline.
3. Check whether the expected source appears in the retrieved context.
4. Check citation correctness and answer faithfulness.
5. Record latency, token usage, and failures.
6. Store results with the model, prompt, embedding, and code version.

The initial dataset is intentionally synthetic. It must be expanded before making performance claims.
