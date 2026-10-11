"""Retrieve web excerpts with Parallel Search MCP and rerank them with a CrossEncoder.

See README.md in this directory for installation and usage.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from datetime import timedelta

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult

from sentence_transformers import CrossEncoder

MCP_URL = "https://search.parallel.ai/mcp"
USER_AGENT = "sentence-transformers/parallel-search-reranking-example"


def search_documents(result: CallToolResult) -> list[dict[str, str]]:
    """Extract one document per result, retaining the URL for reranking output."""
    if result.isError:
        raise RuntimeError("Parallel web_search returned an error. Check the service response before retrying.")
    data = result.structuredContent
    if data is None:
        # MCP servers may return JSON in text blocks rather than structuredContent.
        data = json.loads("".join(block.text for block in result.content if block.type == "text"))
    documents = []
    for item in data["results"]:
        text = "\n\n".join(excerpt for excerpt in item.get("excerpts", []) if excerpt.strip())
        if text:
            documents.append({"text": text, "url": item["url"], "title": item.get("title") or item["url"]})
    return documents


async def search(query: str) -> list[dict[str, str]]:
    """Search anonymously using the MCP SDK's Streamable HTTP transport."""
    async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}, timeout=60) as http:
        async with streamable_http_client(MCP_URL, http_client=http) as (read, write, _):
            async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=60)) as session:
                await session.initialize()
                tools = await session.list_tools()
                if not any(tool.name == "web_search" for tool in tools.tools):
                    raise RuntimeError("The MCP server does not expose web_search.")
                result = await session.call_tool(
                    "web_search",
                    {
                        "objective": query,
                        "search_queries": [query],
                        "session_id": str(uuid.uuid4()),
                    },
                )
                return search_documents(result)


def rerank(query: str, documents: list[dict[str, str]], model: CrossEncoder, top_k: int) -> list[dict]:
    """Rank excerpts and map each corpus_id back to its original source."""
    if not documents:
        return []
    ranks = model.rank(query, [document["text"] for document in documents], top_k=top_k)
    return [{**documents[rank["corpus_id"]], "score": float(rank["score"])} for rank in ranks]


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="A web search query (prefer 3-6 words).")
    parser.add_argument("--top-k", type=positive_int, default=5, help="Number of ranked results to print.")
    parser.add_argument("--model", default="cross-encoder/ms-marco-MiniLM-L6-v2")
    args = parser.parse_args()

    documents = asyncio.run(search(args.query))
    if not documents:
        print("No search results with excerpts were returned.")
        return
    model = CrossEncoder(args.model)
    print(f"Query: {args.query}\nReranking {len(documents)} web results")
    for index, document in enumerate(rerank(args.query, documents, model, args.top_k), start=1):
        print(f"\n{index}. {document['title']} (score: {document['score']:.3f})")
        print(document["url"])
        print(document["text"])


if __name__ == "__main__":
    main()
