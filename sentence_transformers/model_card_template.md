---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
{{ card_data }}
---

# {{ model_name if model_name else "Sentence Transformer model" }}

This is a [sentence-transformers](https://www.SBERT.net) model{% if base_model %} finetuned from [{{ base_model }}](https://huggingface.co/{{ base_model }}){% else %} trained{% endif %}{% if train_datasets | selectattr("name") | list %} on {% if train_datasets | selectattr("name") | map(attribute="name") | join(", ") | length > 200 %}{{ train_datasets | length }}{% else %}the {% for dataset in (train_datasets | selectattr("name")) %}{% if dataset.id %}[{{ dataset.name if dataset.name else dataset.id }}](https://huggingface.co/datasets/{{ dataset.id }}){% else %}{{ dataset.name }}{% endif %}{% if not loop.last %}{% if loop.index == (train_datasets | selectattr("name") | list | length - 1) %} and {% else %}, {% endif %}{% endif %}{% endfor %}{% endif %} dataset{{"s" if train_datasets | selectattr("name") | list | length > 1 else ""}}{% endif %}. It maps sentences & paragraphs to a {{ output_dimensionality }}-dimensional dense vector space and can be used for {{ task_name }}.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
{% if base_model -%}
    {%- if base_model_revision -%}
    - **Base model:** [{{ base_model }}](https://huggingface.co/{{ base_model }}) <!-- at revision {{ base_model_revision }} -->
    {%- else -%}
    - **Base model:** [{{ base_model }}](https://huggingface.co/{{ base_model }})
    {%- endif -%}
{%- else -%}
    <!-- - **Base model:** [Unknown](https://huggingface.co/unknown) -->
{%- endif %}
- **Maximum Sequence Length:** {{ model_max_length }} tokens
- **Output Dimensionality:** {{ output_dimensionality }} dimensions
- **Similarity Function:** {{ similarity_fn_name }}
{% if train_datasets | selectattr("name") | list -%}
    - **Training Dataset{{"s" if train_datasets | selectattr("name") | list | length > 1 else ""}}:**
    {%- for dataset in (train_datasets | selectattr("name")) %}
        {%- if dataset.id %}
    - [{{ dataset.name if dataset.name else dataset.id }}](https://huggingface.co/datasets/{{ dataset.id }})
        {%- else %}
    - {{ dataset.name }}
        {%- endif %}
    {%- endfor %}
{%- else -%}
    <!-- - **Training Dataset:** Unknown -->
{%- endif %}
{% if language -%}
    - **Language{{"s" if language is not string and language | length > 1 else ""}}:**
    {%- if language is string %} {{ language }}
    {%- else %} {% for lang in language -%}
            {{ lang }}{{ ", " if not loop.last else "" }}
        {%- endfor %}
    {%- endif %}
{%- else -%}
    <!-- - **Language:** Unknown -->
{%- endif %}
{% if license -%}
    - **License:** {{ license }}
{%- else -%}
    <!-- - **License:** Unknown -->
{%- endif %}

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
{{ model_string }}
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```
{% if not ir_model %}
Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the {{ hf_emoji }} Hub
model = SentenceTransformer("{{ model_id | default('sentence_transformers_model_id', true) }}")
# Run inference
sentences = [
{%- for text in (predict_example or ["The weather is lovely today.", "It's so sunny outside!", "He drove to the stadium."]) %}
    {{ "%r" | format(text) }},
{%- endfor %}
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [{{ (predict_example or ["The weather is lovely today.", "It's so sunny outside!", "He drove to the stadium."]) | length}}, {{ output_dimensionality | default(1024, true) }}]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
{% if similarities %}print(similarities)
{{ similarities }}{% else %}print(similarities.shape)
# [{{ (predict_example or ["The weather is lovely today.", "It's so sunny outside!", "He drove to the stadium."]) | length}}, {{ (predict_example or ["The weather is lovely today.", "It's so sunny outside!", "He drove to the stadium."]) | length}}]{% endif %}
```
{% else %}
Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the {{ hf_emoji }} Hub
model = SentenceTransformer("{{ model_id | default('sentence_transformers_model_id', true) }}")
# Run inference
queries = [
    {{ predict_example | first | tojson }},
]
documents = [
{%- for text in predict_example[1:] %}
    {{ "%r" | format(text) }},
{%- endfor %}
]
query_embeddings = model.encode_query(queries)
document_embeddings = model.encode_document(documents)
print(query_embeddings.shape, document_embeddings.shape)
# [1, {{ output_dimensionality | default(1024, true) }}] [{{ (predict_example | length) - 1 }}, {{ output_dimensionality | default(1024, true) }}]

# Get the similarity scores for the embeddings
similarities = model.similarity(query_embeddings, document_embeddings)
{% if similarities %}print(similarities)
{{ similarities }}{% else %}print(similarities.shape)
# [1, {{ (predict_example | length) - 1 }}]{% endif %}
```
{% endif %}
<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->
{% if eval_metrics %}
## Evaluation

### Metrics
{% for metrics in eval_metrics %}
#### {{ metrics.description }}
{% if metrics.dataset_name %}
* Dataset{% if metrics.dataset_name is not string and metrics.dataset_name | length > 1 %}s{% endif %}: {% if metrics.dataset_name is string -%}
        `{{ metrics.dataset_name }}`
    {%- else -%}
        {%- for name in metrics.dataset_name -%}
            `{{ name }}`
            {%- if not loop.last -%}
                {%- if loop.index == metrics.dataset_name | length - 1 %} and {% else -%}, {% endif -%}
            {%- endif -%}
        {%- endfor -%}
    {%- endif -%}
{%- endif %}
* Evaluated with {% if metrics.class_name.startswith("sentence_transformers.") %}[<code>{{ metrics.class_name.split(".")[-1] }}</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.{{ metrics.class_name.split(".")[-1] }}){% else %}<code>{{ metrics.class_name }}</code>{% endif %}{% if metrics.config_code %} with these parameters:
{{ metrics.config_code }}{% endif %}

{{ metrics.table }}
{%- endfor %}{% endif %}
<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details
{% for dataset_type, dataset_list in [("training", train_datasets), ("evaluation", eval_datasets)] %}{% if dataset_list %}
### {{ dataset_type.title() }} Dataset{{"s" if dataset_list | length > 1 else ""}}
{% for dataset in dataset_list %}{% if dataset_list | length > 3 %}<details><summary>{{ dataset['name'] or 'Unnamed Dataset' }}</summary>
{% endif %}
#### {{ dataset['name'] or 'Unnamed Dataset' }}
{% if dataset['name'] %}
* Dataset: {% if 'id' in dataset %}[{{ dataset['name'] }}](https://huggingface.co/datasets/{{ dataset['id'] }}){% else %}{{ dataset['name'] }}{% endif %}
{%- if 'revision' in dataset and 'id' in dataset %} at [{{ dataset['revision'][:7] }}](https://huggingface.co/datasets/{{ dataset['id'] }}/tree/{{ dataset['revision'] }}){% endif %}{% endif %}
{% if dataset['size'] %}* Size: {{ "{:,}".format(dataset['size']) }} {{ dataset_type }} samples
{% endif %}* Columns: {% if dataset['columns'] | length == 1 %}{{ dataset['columns'][0] }}{% elif dataset['columns'] | length == 2 %}{{ dataset['columns'][0] }} and {{ dataset['columns'][1] }}{% else %}{{ dataset['columns'][:-1] | join(', ') }}, and {{ dataset['columns'][-1] }}{% endif %}
{% if dataset['stats_table'] %}* Approximate statistics based on the first {{ [dataset['size'], 1000] | min }} samples:
{{ dataset['stats_table'] }}{% endif %}{% if dataset['examples_table'] %}* Samples:
{{ dataset['examples_table'] }}{% endif %}* Loss: {% if dataset["loss"]["fullname"].startswith("sentence_transformers.") %}[<code>{{ dataset["loss"]["fullname"].split(".")[-1] }}</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#{{ dataset["loss"]["fullname"].split(".")[-1].lower() }}){% else %}<code>{{ dataset["loss"]["fullname"] }}</code>{% endif %}{% if "config_code" in dataset["loss"] %} with these parameters:
{{ dataset["loss"]["config_code"] }}{% endif %}
{% if dataset_list | length > 3 %}</details>
{% endif %}{% endfor %}{% endif %}{% endfor -%}

{% if all_hyperparameters %}
### Training Hyperparameters
{% if non_default_hyperparameters -%}
#### Non-Default Hyperparameters

{% for name, value in non_default_hyperparameters.items() %}- `{{ name }}`: {{ value }}
{% endfor %}{%- endif %}
#### All Hyperparameters
<details><summary>Click to expand</summary>

{% for name, value in all_hyperparameters.items() %}- `{{ name }}`: {{ value }}
{% endfor %}
</details>
{% endif %}

{%- if eval_lines %}
### Training Logs
{% if hide_eval_lines %}<details><summary>Click to expand</summary>

{% endif -%}
{{ eval_lines }}{% if explain_bold_in_eval %}
* The bold row denotes the saved checkpoint.{% endif %}
{%- if hide_eval_lines %}
</details>{% endif %}
{% endif %}

{%- if co2_eq_emissions %}
### Environmental Impact
Carbon emissions were measured using [CodeCarbon](https://github.com/mlco2/codecarbon).
- **Energy Consumed**: {{ "%.3f"|format(co2_eq_emissions["energy_consumed"]) }} kWh
- **Carbon Emitted**: {{ "%.3f"|format(co2_eq_emissions["emissions"] / 1000) }} kg of CO2
- **Hours Used**: {{ co2_eq_emissions["hours_used"] }} hours

### Training Hardware
- **On Cloud**: {{ "Yes" if co2_eq_emissions["on_cloud"] else "No" }}
- **GPU Model**: {{ co2_eq_emissions["hardware_used"] or "No GPU used" }}
- **CPU Model**: {{ co2_eq_emissions["cpu_model"] }}
- **RAM Size**: {{ "%.2f"|format(co2_eq_emissions["ram_total_size"]) }} GB
{% endif %}
### Framework Versions
- Python: {{ version["python"] }}
- Sentence Transformers: {{ version["sentence_transformers"] }}
- Transformers: {{ version["transformers"] }}
- PyTorch: {{ version["torch"] }}
- Accelerate: {{ version["accelerate"] }}
- Datasets: {{ version["datasets"] }}
- Tokenizers: {{ version["tokenizers"] }}

## Citation

### BibTeX
{% for loss_name, citation in citations.items() %}
#### {{ loss_name }}
```bibtex
{{ citation | trim }}
```
{% endfor %}
<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->
