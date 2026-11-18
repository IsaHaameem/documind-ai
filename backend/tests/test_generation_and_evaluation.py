import json
from uuid import uuid4

import httpx
import pytest

from documind.config import Settings
from documind.evaluation import demo_bytes, demo_cases, metrics
from documind.generation import DeepSeek, GenerationError, generate, rewrite
from documind.retrieval import Hit


class Reply:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def json_response(self, system, payload, *, max_tokens=1300):
        self.calls.append((system, payload))
        return json.dumps(self.payload)


def hit(text):
    return Hit(uuid4(), uuid4(), "returns.pdf", "Page 4", 4, text, 0.81)


def test_grounded_answer_resolves_to_real_source():
    passage = hit("Requests must be filed within 30 calendar days.")
    provider = Reply({"insufficient_context": False, "claims": [
        {"statement": "You have 30 calendar days.", "evidence": [
            {"source": 1, "quote": "within 30 calendar days"}
        ]}
    ]})
    answer = generate(provider, "What is the deadline?", [passage])
    assert answer.text == "You have 30 calendar days. [1]"
    assert answer.citations[0].chunk_id == passage.chunk_id
    assert answer.citations[0].page == 4
    assert answer.citations[0].quote == "within 30 calendar days"


@pytest.mark.parametrize(("source", "quote"), [(2, "within 30 calendar days"), (1, "within two months")])
def test_fabricated_evidence_is_rejected(source, quote):
    provider = Reply({"insufficient_context": False, "claims": [
        {"statement": "You have time.", "evidence": [{"source": source, "quote": quote}]}
    ]})
    result = generate(provider, "What is the deadline?", [hit("File within 30 calendar days.")])
    assert result.insufficient_context
    assert not result.citations


def test_no_results_abstains_without_provider_and_rewrite_uses_followup_history():
    provider = Reply({"query": "What is the refund deadline?"})
    assert generate(provider, "Unknown answer?", []).insufficient_context
    assert provider.calls == []
    assert rewrite(provider, "And the deadline?", ["How do refunds work?"]) == "What is the refund deadline?"


def test_deepseek_provider_contract_and_timeout():
    provider = DeepSeek(Settings(_env_file=None, deepseek_api_key="test-only-provider-token"))
    provider.client.close()

    def completion(request):
        assert request.url.path == "/chat/completions"
        body = json.loads(request.content)
        assert body["thinking"] == {"type": "disabled"}
        assert body["response_format"] == {"type": "json_object"}
        return httpx.Response(200, json={"choices": [{
            "finish_reason": "stop", "message": {"content": '{"query":"refund deadline"}'}
        }]})

    with httpx.Client(base_url="https://api.deepseek.com", transport=httpx.MockTransport(completion)) as client:
        provider.client = client
        assert json.loads(provider.json_response("Return JSON.", {}))["query"] == "refund deadline"

    def timed_out(request):
        raise httpx.ReadTimeout("test timeout", request=request)

    with httpx.Client(base_url="https://api.deepseek.com", transport=httpx.MockTransport(timed_out)) as client:
        provider.client = client
        with pytest.raises(GenerationError, match="timed out"):
            provider.json_response("Return JSON.", {})


def test_retrieval_metrics_and_demo_ground_truth():
    demo = demo_bytes().decode()
    assert all(passage in demo for case in demo_cases() for passage in case.relevant_passages)
    results = metrics([hit("irrelevant"), hit("One correct passage.")],
                      ["One correct passage.", "A second relevant passage."])
    assert results == {"recall_at_k": 0.5, "mrr": 0.5, "context_precision": 0.5, "context_recall": 0.5}
