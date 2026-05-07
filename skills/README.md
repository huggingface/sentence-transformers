# Sentence-Transformers Agent Skill

Tool-neutral [Agent Skill](https://agentskills.io) for training models with the
`sentence-transformers` library. The single `SKILL.md` + references + runnable
example scripts in [`train-sentence-transformers/`](train-sentence-transformers/)
covers all three architectures and lets any compatible coding agent drive a
training run end-to-end (model selection, hard-negative mining, loss / evaluator
choice, training, evaluation, Hub publishing):

| Architecture | What it does |
|---|---|
| `SentenceTransformer` (bi-encoder) | Fixed-dimension dense vectors for semantic similarity, retrieval, clustering, classification. |
| `CrossEncoder` (reranker) | Pair scoring for two-stage retrieval / pairwise classification. |
| `SparseEncoder` (SPLADE) | Sparse vectors over the vocabulary for learned-sparse retrieval (Elasticsearch / OpenSearch / Lucene-compatible). |

Loss / evaluator / example references inside the skill are split per
architecture (`losses_sentence_transformer.md`, `losses_cross_encoder.md`,
`losses_sparse_encoder.md`, etc.); `SKILL.md` directs the agent to load only
the ones matching the user's task.

## Install

You don't need to clone this repo to use the skill. It's published to the
[`huggingface/skills`](https://github.com/huggingface/skills) marketplace and
installable via your agent's standard plugin / skill mechanism.

### Hugging Face CLI (works for Claude Code, Codex, Gemini CLI, Cursor)

```bash
hf skills add train-sentence-transformers
```

This installs the skill into `~/.agents/skills/` (the cross-tool standard
location). To later refresh it to the latest version:

```bash
hf skills update
```

### Claude Code plugin marketplace

```text
/plugin marketplace add huggingface/skills
/plugin install train-sentence-transformers@huggingface/skills
```

### Other agents

[`huggingface/skills`](https://github.com/huggingface/skills) auto-publishes
to the Cursor Marketplace, the Codex Plugins Directory, and the Gemini CLI
extensions. See its README for the per-tool install command.

## Using the skill

Once installed, just mention the task naturally — the agent loads `SKILL.md`
automatically based on its `description` frontmatter:

> "Train a multilingual sentence-transformer on my parallel corpus."
> "Fine-tune a cross-encoder reranker on `(question, answer)` pairs from my
> dataset, mine hard negatives, and push to my Hub repo."
> "Train a SPLADE model from `naver/splade-v3` on domain data and evaluate
> sparsity."

## Local development

If you want to iterate on the skill from a local clone of this repo (so your
agent picks up edits instantly without going through `hf skills add`),
symlink the skill subdirectory into your agent's standard skill location.
Symlinking the individual subdir (rather than the whole `skills/` directory)
lets it coexist with any other skills you already have installed there.
`.claude/` and `.agents/` are gitignored, so the link won't be tracked.

**macOS / Linux:**

```bash
mkdir -p .claude/skills .agents/skills
ln -sfn "$(pwd)/skills/train-sentence-transformers" .claude/skills/train-sentence-transformers
ln -sfn "$(pwd)/skills/train-sentence-transformers" .agents/skills/train-sentence-transformers
```

**Windows** (directory junctions — no admin or Developer Mode needed):

```cmd
if not exist .claude\skills mkdir .claude\skills
mklink /J ".claude\skills\train-sentence-transformers" "%cd%\skills\train-sentence-transformers"

if not exist .agents\skills mkdir .agents\skills
mklink /J ".agents\skills\train-sentence-transformers" "%cd%\skills\train-sentence-transformers"
```

After this, edits to `SKILL.md` / any reference / any script under
`skills/train-sentence-transformers/` are immediately visible to your agent on
the next skill invocation. If you only use one tool, drop the corresponding
line for the other.

## Maintenance

The skill is mirrored automatically to `huggingface/skills` on every
release tag via [.github/workflows/sync-skills.yml](../.github/workflows/sync-skills.yml).
The canonical source lives here in the `sentence-transformers` repo.
