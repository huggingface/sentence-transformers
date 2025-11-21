"""
This example demonstrates how to use the CrossEncoder with instruction-tuned models like Qwen-reranker or BGE-reranker.
The new `prompt_template` and `prompt_template_kwargs` arguments in the `predict` and `rank` methods allow for
flexible and dynamic formatting of the input for such models.

This script covers three main scenarios:
1.  Ranking without any template (baseline).
2.  Ranking with a `prompt_template` provided at runtime.
3.  Ranking with a dynamic `instruction` passed via `prompt_template_kwargs`.

Finally, it provides a guide on how to set a default prompt template in the model's `config.json`.
"""

from sentence_transformers.cross_encoder import CrossEncoder

# We use a Qwen Reranker model here. In a real-world scenario, this could also be
# an instruction-tuned model like 'BAAI/bge-reranker-large'.
model = CrossEncoder("tomaarsen/Qwen3-Reranker-0.6B-seq-cls", trust_remote_code=True)
model.model.config.pad_token_id = model.tokenizer.pad_token_id

query = "What is the capital of China?"
documents = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
]

# First, we create the sentence pairs for the query and all documents
sentence_pairs = [[query, doc] for doc in documents]

print("--- 1. Reranking without any template (Incorrect Usage of Qwen3 Reranker) ---")
# The model receives the plain query and document pairs.
baseline_scores = model.predict(sentence_pairs, convert_to_numpy=True)
scored_docs = sorted(zip(baseline_scores, documents), key=lambda x: x[0], reverse=True)

print("Query:", query)
for score, doc in scored_docs:
    print(f"{score:.4f}\t{doc}")

print("\n\n--- 2. Reranking with a runtime prompt_template ---")
# The query and document are formatted using the template before being passed to the model.
# This changes the input text and thus the resulting scores.
prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
instruction = "Given a web search query, retrieve relevant passages that answer the query"
query_template = f"{prefix}<Instruct>: {instruction}\n<Query>: {{query}}\n"
document_template = f"<Document>: {{document}}{suffix}"

template = query_template + document_template
template_scores = model.predict(sentence_pairs, prompt_template=template)
scored_docs_template = sorted(zip(template_scores, documents), key=lambda x: x[0], reverse=True)

print("Using template:", template)
print("Query:", query)
for score, doc in scored_docs_template:
    print(f"{score:.4f}\t{doc}")
# The scores will be different from the baseline because the model processes a different text.

print("\n\n--- 3. Reranking with a dynamic instruction ---")
# This is useful for models that expect a specific instruction.
# The instruction can be changed at runtime.
prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
instruct_template = f"{prefix}<Instruct>: {{instruction}}\n<Query>: {{query}}\n<Document>: {{document}}{suffix}"
instruct_kwargs_1 = {"instruction": "Given a query, find the most relevant document."}
instruct_kwargs_2 = {"instruction": "Given a question, find the incorrect answer."}  # Misleading instruction

print(f"Using template: {instruct_template}")
print(f"With instruction 1: '{instruct_kwargs_1['instruction']}'")
instruction_scores_1 = model.predict(
    sentence_pairs, prompt_template=instruct_template, prompt_template_kwargs=instruct_kwargs_1
)
scored_docs_instruct_1 = sorted(zip(instruction_scores_1, documents), key=lambda x: x[0], reverse=True)
for score, doc in scored_docs_instruct_1:
    print(f"{score:.4f}\t{doc}")

print(f"\nWith instruction 2: '{instruct_kwargs_2['instruction']}'")
instruction_scores_2 = model.predict(
    sentence_pairs, prompt_template=instruct_template, prompt_template_kwargs=instruct_kwargs_2
)
scored_docs_instruct_2 = sorted(zip(instruction_scores_2, documents), key=lambda x: x[0], reverse=True)
for score, doc in scored_docs_instruct_2:
    print(f"{score:.4f}\t{doc}")
# The scores for instruction 1 and 2 will likely differ, as the instruction text changes the input.

# --- 4. Guide: Setting a Default Prompt Template in config.json ---
#
# If you are a model creator or want to use a specific prompt format consistently
# without passing it in every `rank` or `predict` call, you can set a default
# template in the model's `config.json` file.
#
# Step 1: Save your base model to a directory.
#
#   from sentence_transformers import CrossEncoder
#   import json
#
#   model = CrossEncoder("your-base-model-name")
#   save_path = "path/to/your-instruct-model"
#   model.save(save_path)
#
# Step 2: Modify the `config.json` in the saved directory.
#   Add the "prompt_template" and "prompt_template_kwargs" keys to the
#   "sentence_transformers" dictionary.
#
#   // path/to/your-instruct-model/config.json
#   {
#     ...
#     "sentence_transformers": {
#       "version": "3.0.0.dev0",
#       "prompt_template": "Instruct: {instruction}\nQuery: {query}\nDocument: {document}",
#       "prompt_template_kwargs": {
#         "instruction": "Given a query, find the most relevant document."
#       }
#     },
#     ...
#   }
#
# Step 3: Load the model from the modified path.
#   It will now use the default template automatically.
#
#   instruct_model = CrossEncoder(save_path, trust_remote_code=True)
#   sentence_pairs = [[query, doc] for doc in documents]
#   scores = instruct_model.predict(sentence_pairs)
#
#   # This call is now equivalent to calling the original model with the full template arguments:
#   # original_model.predict(sentence_pairs,
#   #                     prompt_template="Instruct: {instruction}\nQuery: {query}\nDocument: {document}",
#   #                     prompt_template_kwargs={"instruction": "Given a query, find the most relevant document."})
#
# You can still override the default template by passing arguments at runtime:
#
#   # This will use the new instruction, overriding the default one.
#   scores_new_instruction = instruct_model.predict(
#       sentence_pairs,
#       prompt_template_kwargs={"instruction": "Find the answer to the question."}
#   )
