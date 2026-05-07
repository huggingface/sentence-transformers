# Sentence-Transformers Agent Skill

This is a tool-neutral [Agent Skill](https://agentskills.io) for training and fine-tuning `sentence-transformers` models. It covers model selection, hard-negative mining, loss / evaluator choice, training, evaluation, and Hub publishing across all three archetypes:

| Archetype | What it does |
|---|---|
| `SentenceTransformer` (bi-encoder) | Fixed-dimension dense vectors for semantic similarity, retrieval, clustering, classification. |
| `CrossEncoder` (reranker) | Pair scoring for two-stage retrieval / pairwise classification. |
| `SparseEncoder` (SPLADE) | Sparse vectors over the vocabulary for learned-sparse retrieval. |

Loss / evaluator / example references inside the skill are split per archetype (`losses_sentence_transformer.md`, `losses_cross_encoder.md`, `losses_sparse_encoder.md`, etc.). `SKILL.md` directs the agent to load only the ones matching the user's task.

## Install

You don't need to clone this repo to use the skill. It's published to [`huggingface/skills`](https://github.com/huggingface/skills).

### Codex, Cursor, Gemini CLI, OpenCode (Agent Skills standard)

```bash
hf skills add train-sentence-transformers              # ./.agents/skills/<name>
hf skills add train-sentence-transformers --global     # ~/.agents/skills/<name>
```

Any agent that follows the [Agent Skills](https://agentskills.io) standard auto-discovers skills under `.agents/skills/`. `hf skills update` refreshes installed skills to the latest upstream commit.

### Claude Code

Claude Code reads from `.claude/skills/`, so the `--claude` flag also symlinks the install there:

```bash
hf skills add train-sentence-transformers --claude
```

## Using the skill

Once installed, just mention the task naturally, and the agent loads `SKILL.md` automatically based on its `description` frontmatter:

> "Train a multilingual sentence-transformer on my parallel corpus."
> "Fine-tune a cross-encoder reranker on `(question, answer)` pairs from my dataset, mine hard negatives, and push to my Hub repo."
> "Train a SPLADE model from `naver/splade-v3` on domain data and evaluate sparsity."

## Local development

If you want to iterate on the skill from a local clone of this repo (so your agent picks up edits instantly without going through `hf skills add`), symlink the skill subdirectory into your agent's standard skill location. Symlinking the individual subdir (rather than the whole `skills/` directory) lets it coexist with any other skills you already have installed there.

**macOS / Linux:**

```bash
mkdir -p .agents/skills
ln -sfn "$(pwd)/skills/train-sentence-transformers" .agents/skills/train-sentence-transformers
```

**Windows:**

```cmd
if not exist .agents\skills mkdir .agents\skills
mklink /J ".agents\skills\train-sentence-transformers" "%cd%\skills\train-sentence-transformers"
```

For Claude Code, repeat with `.claude/skills/` as the target.

After this, edits to `SKILL.md` / any reference / any script under `skills/train-sentence-transformers/` are immediately visible to your agent on the next skill invocation. If you only use one tool, drop the corresponding line for the other.

## Maintenance

The skill is mirrored automatically to `huggingface/skills` on every release tag via [.github/workflows/sync-skills.yml](../.github/workflows/sync-skills.yml). The canonical source lives here in the `sentence-transformers` repo.
