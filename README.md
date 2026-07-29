# Causal Inference Explorer

This repository contains the anonymous code package for the paper submission. It provides a Streamlit/FastAPI app for evaluating causal-direction changes in bivariate datasets with two regimes. The app compares LLM-based causal reasoning with data-driven causal discovery methods.

## What Is Included

- Processed bivariate datasets and metadata in `DATA/`
- Streamlit/FastAPI application code in `app/`
- Implementations/wrappers for ROCHE, LOCI, LCUBE, and LiNGAM baselines
- Batch experiment panels and result-export utilities
- Prompt-chain code for LLM experiments
- Docker files for reproducible local execution
- Selected saved CSV outputs used during development and validation

Raw source-material folders, caches, local API keys, and author-identifying files are not included.

## Quick Start

### 1. Configure The API Key

LLM runs use OpenRouter. Classical causal-method runs and dataset inspection do not need an API key.

```bash
cp .env.template .env
```

Edit `.env`:

```bash
OPENROUTER_API_KEY=your-openrouter-api-key-here
OPENROUTER_HTTP_REFERER=http://localhost
OPENROUTER_X_TITLE=Causal Inference Explorer
```

### 2. Build And Run

```bash
docker compose up --build
```

If your Docker installation uses the older command form, use:

```bash
docker-compose up --build
```

### 3. Open The App

Visit:

```text
http://localhost:8501
```

The FastAPI backend is exposed inside the Docker setup and is used by the Streamlit interface.

## Main Workflows

### Inspect A Dataset

Use the dataset panel to select a pair. The app shows the bivariate data, the threshold variable, the threshold value, and the two resulting regimes.

### Run Classical Causal Methods

Use the causal-analysis panel to run the available data-driven methods per regime. The app compares predicted causal direction against the ground-truth metadata in `DATA/pairmeta_with_ground_truth.txt`.

### Run LLM Experiments

Use the LLM panel or batch panel to run the prompt chain through OpenRouter-hosted models. The main experiment configurations are:

- `raw_named_hint_explicit`: raw data with variable names
- `raw_anon_hint_explicit`: raw data with anonymized variable labels
- `segmented_named_hint_explicit`: threshold-segmented regimes with variable names
- `segmented_anon_hint_explicit`: threshold-segmented regimes with anonymized variable labels

The prompt-chain source is:

```text
app/utils/prompts.py
```

The LLM model and sampling configuration is in:

```text
app/utils/llm_config.py
```

## Sampling Policy

For batch LLM experiments:

- Repeated calls use deterministic seeds starting at 42.
- Raw-data configurations pass the full pair when it is below the configured row cap.
- Larger raw-data pairs are sampled directly from the full pair for each repeated run.
- Segmented-regime configurations split by the metadata threshold first.
- Larger regimes are sampled after segmentation.
- Repeated outputs are aggregated by majority vote per regime before scoring.

The row preview used by the UI is separate from the batch-experiment sampling path.

## Repository Structure

```text
.
|-- app/                  # Streamlit/FastAPI application
|   |-- api/              # API endpoints and task handling
|   |-- components/       # UI panels
|   `-- utils/            # Data loading, prompts, LLM client, scoring
|-- DATA/                 # Processed datasets and pair metadata
|   `-- custom_pairs/     # Bivariate pairs used by the app
|-- LCUBE/                # LCUBE implementation/wrapper
|-- LINGAM/               # LiNGAM wrapper
|-- LOCI/                 # LOCI implementation/wrapper
|-- ROCHE/                # ROCHE implementation/wrapper
|-- helpers/              # Shared utilities
|-- scripts/              # Dataset-generation and capping scripts
|-- results/              # Output folders and selected CSV outputs
|-- Dockerfile
|-- docker-compose.yml
|-- requirements.runtime.txt
`-- test_ground_truth.py
```

## Data And Metadata

Dataset files live in:

```text
DATA/custom_pairs/
```

Ground-truth threshold and direction metadata lives in:

```text
DATA/pairmeta_with_ground_truth.txt
```

Each metadata row specifies the dataset identifier, column indices, threshold variable, threshold value, and causal direction in each regime.

## Reproducing The Validation Check

After building the Docker image, run:

```bash
docker compose exec causal-app python test_ground_truth.py
```

Expected outcome: the script completes without ground-truth lookup errors.

## Environment Variables

| Variable | Required | Description |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | Required for LLM runs | OpenRouter API key |
| `OPENROUTER_HTTP_REFERER` | No | Optional HTTP referer header |
| `OPENROUTER_X_TITLE` | No | Optional title header |

## External Methods

This app uses or wraps the following causal-discovery methods for comparison:

- LOCI, Location-Scale Noise Models: <https://github.com/AlexImmer/loci>
- ROCHE, robust causal discovery with Student's t-distribution: <https://github.com/quangdzuytran/ROCHE>
- LCUBE, MDL-based causal discovery with cubic splines: <https://github.com/suzi216/LCUBE>
- LiNGAM: <https://github.com/cdt15/lingam>

## Notes For Reviewers

This package is intended for anonymous academic review. It contains processed datasets and runnable code, but no private API keys. LLM experiments require the reviewer to provide their own OpenRouter key.
