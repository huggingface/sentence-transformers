from __future__ import annotations

import os

# A chat template in the style of instruct backbones: branches on the chat roles only, so the
# query/document roles that pairs are mapped to render to nothing.
GENERIC_INSTRUCT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}<|user|>\n{{ message['content'] }}\n{% endif %}"
    "{% endfor %}"
)
# A chat template in the style of published rerankers: selects the query/document roles explicitly.
RERANKER_TEMPLATE = (
    '<Query>: {{ messages | selectattr("role", "eq", "query") | map(attribute="content") | first }}\n'
    '<Document>: {{ messages | selectattr("role", "eq", "document") | map(attribute="content") | first }}'
)


def is_ci() -> bool:
    """
    Check if the code is running in a Continuous Integration (CI) environment.
    This is determined by checking for the presence of certain environment variables.
    """
    return "GITHUB_ACTIONS" in os.environ
