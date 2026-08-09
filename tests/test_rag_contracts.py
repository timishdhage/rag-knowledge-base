import json
from pathlib import Path


DATASET = Path(__file__).parents[1] / "evaluation" / "questions.json"
REQUIRED_FIELDS = {
    "id",
    "question",
    "expected_source",
    "expected_answer_points",
    "should_refuse",
}


def test_evaluation_dataset_is_valid_json():
    records = json.loads(DATASET.read_text())
    assert isinstance(records, list)
    assert records


def test_evaluation_records_have_stable_contract():
    records = json.loads(DATASET.read_text())
    ids = []
    for record in records:
        assert REQUIRED_FIELDS.issubset(record)
        assert record["id"] not in ids
        assert record["question"]
        assert isinstance(record["expected_answer_points"], list)
        assert isinstance(record["should_refuse"], bool)
        ids.append(record["id"])


def test_refusal_cases_have_no_expected_source():
    records = json.loads(DATASET.read_text())
    for record in records:
        if record["should_refuse"]:
            assert record["expected_source"] is None
