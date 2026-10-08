"""Grounded extractive answer composer + LLM adapter stub.

The demo answer path is **extractive**: it composes an answer from the
retrieved chunks and cites the source doc(s). No hallucinations by design —
every claim in the answer is traceable to a cited chunk.

``LLMAdapter`` is the production seam. The local implementation raises;
the documented Bedrock path (commented implementation below) calls
``bedrock-runtime`` Converse API with the retrieved chunks as context.
No credentials are needed for the demo — the Bedrock path is documented,
not faked.
"""
from __future__ import annotations


def compose_answer(search_result: dict, max_chars: int = 600) -> dict:
    """Build an extractive answer from hybrid retrieval results.

    Results are document-ranked (score-aggregated); the answer is extracted
    from the best document's top chunk and every answer carries doc citations.
    """
    results = search_result["results"]
    if not results:
        return {"answer": "No relevant runbooks found.", "citations": [],
                "latency_ms": search_result["latency_ms"]}
    best = results[0]
    lead_chunk = best["chunks"][0]

    answer = lead_chunk["text"][:max_chars].rstrip()
    citations = [
        {
            "doc_id": r["doc_id"],
            "title": r["title"],
            "section": r["chunks"][0]["section"],
            "score": r["score"],
        }
        for r in results[:3]
    ]
    return {
        "answer": answer,
        "citations": citations,
        "latency_ms": search_result["latency_ms"],
        "retrieved_docs": len(results),
    }


class LLMAdapter:
    """Pluggable generative layer. Local demo: extractive only."""

    def complete(self, query: str, context_chunks: list[dict]) -> str:
        raise NotImplementedError(
            "Generative completion is disabled in the local demo. "
            "Use compose_answer() for grounded extractive answers, or "
            "BedrockLLMAdapter for the production path."
        )


# ---------------------------------------------------------------------------
# Production path (documented, not executed in the demo):
#
#   import boto3
#
#   class BedrockLLMAdapter(LLMAdapter):
#       """Generate answers with Amazon Bedrock (e.g. Claude) grounded on
#       retrieved chunks. Requires AWS credentials with bedrock:InvokeModel."""
#       def __init__(self, model_id="anthropic.claude-3-haiku-20240307-v1:0",
#                    region="us-east-1"):
#           self.client = boto3.client("bedrock-runtime", region_name=region)
#           self.model_id = model_id
#       def complete(self, query, context_chunks):
#           context = "\n\n".join(
#               f"[{c['doc_id']} | {c['section']}] {c['text']}"
#               for c in context_chunks)
#           response = self.client.converse(
#               modelId=self.model_id,
#               messages=[{"role": "user", "content": [{"text":
#                   f"Answer using ONLY the context below. Cite [doc_id] for "
#                   f"every claim.\n\nContext:\n{context}\n\nQuestion: {query}"}]}],
#               inferenceConfig={"maxTokens": 400, "temperature": 0.1})
#           return response["output"]["message"]["content"][0]["text"]
# ---------------------------------------------------------------------------
