from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_api_contract_document_exists():
    content = (ROOT / "docs" / "api-contract.md").read_text()
    assert "## Query endpoint" in content
    assert "## Response shape" in content
    assert "## Error contract" in content


def test_configuration_document_covers_secret_handling():
    content = (ROOT / "docs" / "configuration.md").read_text()
    assert "Secret handling" in content
    assert ".env.example" in content
    assert "least-privilege" in content
