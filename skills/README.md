# Sentence-Transformers Agent Skills

Tool-neutral [Agent Skills](https://agentskills.io) for training models with the
`sentence-transformers` library. Each skill is a self-contained `SKILL.md` +
references + runnable example scripts that any compatible coding agent can
load to drive a training run end-to-end (model selection, hard-negative
mining, loss / evaluator choice, training, evaluation, Hub publishing).

## Skills in this directory

| Skill | What it trains |
|---|---|
| [`train-sentence-transformer`](train-sentence-transformer/) | `SentenceTransformer` (bi-encoder) embedding models for semantic similarity, retrieval, clustering, classification. |
| [`train-cross-encoder`](train-cross-encoder/) | `CrossEncoder` reranker models for two-stage retrieval / pairwise classification. |
| [`train-sparse-encoder`](train-sparse-encoder/) | `SparseEncoder` (SPLADE) models for learned-sparse retrieval (Elasticsearch / OpenSearch / Lucene-compatible). |

Each skill is fully self-contained: its `references/` folder holds both skill-specific docs (loss / evaluator catalogs) and the cross-cutting reference docs that apply to all three (training args, dataset formats, hardware, troubleshooting, prompts, Hugging Face Jobs). The `mine_hard_negatives.py` CLI also ships inside each skill's `scripts/`.

## Install

You don't need to clone this repo to use the skills. They're published to the
[`huggingface/skills`](https://github.com/huggingface/skills) marketplace and
installable via your agent's standard plugin / skill mechanism.

### Hugging Face CLI (works for Claude Code, Codex, Gemini CLI, Cursor)

```bash
hf skills add train-sentence-transformer
hf skills add train-cross-encoder
hf skills add train-sparse-encoder
```

This installs each skill into `~/.agents/skills/` (the cross-tool standard
location). To later refresh installed skills to the latest version:

```bash
hf skills update
```

### Claude Code plugin marketplace

```text
/plugin marketplace add huggingface/skills
/plugin install train-sentence-transformer@huggingface/skills
/plugin install train-cross-encoder@huggingface/skills
/plugin install train-sparse-encoder@huggingface/skills
```

### Other agents

[`huggingface/skills`](https://github.com/huggingface/skills) auto-publishes
to the Cursor Marketplace, the Codex Plugins Directory, and the Gemini CLI
extensions. See its README for the per-tool install command.

## Using a skill

Once installed, just mention the task naturally — the agent loads the matching
`SKILL.md` automatically based on its `description` frontmatter:

> "Train a multilingual sentence-transformer on my parallel corpus."
> "Fine-tune a cross-encoder reranker on `(question, answer)` pairs from my
> dataset, mine hard negatives, and push to my Hub repo."
> "Train a SPLADE model from `naver/splade-v3` on domain data and evaluate
> sparsity."

## Local development

If you want to iterate on a skill from a local clone of this repo (so your
agent picks up edits instantly without going through `hf skills add`),
symlink each skill subdirectory into your agent's standard skill location.
Symlinking individual subdirs (rather than the whole `skills/` directory)
lets these coexist with any other skills you already have installed there.
`.claude/` and `.agents/` are gitignored, so the links won't be tracked.

**macOS / Linux:**

```bash
mkdir -p .claude/skills .agents/skills
ln -sfn "$(pwd)/skills/train-sentence-transformer" .claude/skills/train-sentence-transformer
ln -sfn "$(pwd)/skills/train-cross-encoder"        .claude/skills/train-cross-encoder
ln -sfn "$(pwd)/skills/train-sparse-encoder"       .claude/skills/train-sparse-encoder

ln -sfn "$(pwd)/skills/train-sentence-transformer" .agents/skills/train-sentence-transformer
ln -sfn "$(pwd)/skills/train-cross-encoder"        .agents/skills/train-cross-encoder
ln -sfn "$(pwd)/skills/train-sparse-encoder"       .agents/skills/train-sparse-encoder
```

**Windows** (directory junctions — no admin or Developer Mode needed):

```cmd
if not exist .claude\skills mkdir .claude\skills
mklink /J ".claude\skills\train-sentence-transformer" "%cd%\skills\train-sentence-transformer"
mklink /J ".claude\skills\train-cross-encoder"        "%cd%\skills\train-cross-encoder"
mklink /J ".claude\skills\train-sparse-encoder"       "%cd%\skills\train-sparse-encoder"

if not exist .agents\skills mkdir .agents\skills
mklink /J ".agents\skills\train-sentence-transformer" "%cd%\skills\train-sentence-transformer"
mklink /J ".agents\skills\train-cross-encoder"        "%cd%\skills\train-cross-encoder"
mklink /J ".agents\skills\train-sparse-encoder"       "%cd%\skills\train-sparse-encoder"
```

After this, edits to any `SKILL.md` / script under `skills/<name>/` are
immediately visible to your agent on the next skill invocation. If you only
use one tool, drop the corresponding line for the other.

## Maintenance

These skills are mirrored automatically to `huggingface/skills` on every
release tag via [.github/workflows/sync-skills.yml](../.github/workflows/sync-skills.yml).
The canonical source lives here in the `sentence-transformers` repo.
