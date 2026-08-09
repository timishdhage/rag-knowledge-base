from __future__ import annotations

import pytest

from rag.retrieval import filter_owner_chunks


def test_owner_filter_excludes_other_users() -> None:
    chunks = [
        {'id': 'a', 'owner_id': 'user-a', 'text': 'private A'},
        {'id': 'b', 'owner_id': 'user-b', 'text': 'private B'},
    ]
    result = filter_owner_chunks(chunks, 'user-a')
    assert [chunk['id'] for chunk in result] == ['a']


def test_owner_filter_rejects_blank_owner() -> None:
    with pytest.raises(ValueError, match='owner_id'):
        filter_owner_chunks([], '  ')
