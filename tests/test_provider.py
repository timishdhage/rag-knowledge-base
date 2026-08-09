from src.rag.provider import ModelGateway, OpenAIModelGateway


def test_openai_gateway_implements_generate_contract():
    gateway = OpenAIModelGateway()
    assert hasattr(gateway, "generate")


def test_model_gateway_protocol_is_available():
    assert ModelGateway is not None
