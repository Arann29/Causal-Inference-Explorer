# CODE AND DATA SUPPLEMENT

Can LLMs Rival Data-Driven Methods in Causal Discovery?
A Systematic Comparative Framework

This archive contains the implementation, the processed benchmark datasets,
the data-driven baselines, the LLM prompt-chain code, and selected committed
result files used for the submission. It is self-contained for inspecting the
datasets, running the data-driven causal-discovery baselines, and reproducing
the LLM experiments reported in the paper. The application provides a
Streamlit/FastAPI interface for evaluating regime-dependent causal-direction
changes in bivariate datasets with two regimes, comparing LLM-based causal
reasoning against data-driven causal discovery methods (ROCHE, LOCI, LCUBE,
and LiNGAM).

Raw third-party source-material folders, caches, local API keys, and
author-identifying files are not redistributed.


## QUICK START

The reported results use Docker for a reproducible environment (Python 3.10
inside the image). From this directory:

### 1. Configure The API Key

LLM runs use OpenRouter. Running the data-driven baselines and inspecting the
datasets does not require an API key.

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

The FastAPI backend is exposed inside the Docker setup and is used by the
Streamlit interface.


## CONTENTS

```text
app/                  Streamlit/FastAPI application (UI panels, API, utils)
DATA/                 Processed bivariate datasets and pair metadata
LCUBE/                LCUBE implementation/wrapper
LINGAM/               LiNGAM wrapper
LOCI/                 LOCI implementation/wrapper
ROCHE/                ROCHE implementation/wrapper
helpers/              Shared segmentation and optimization utilities
scripts/              Dataset-generation and sample-capping scripts
results/              Output folders and selected committed CSV outputs
Dockerfile            Container definition (Python 3.10, CPU-only PyTorch)
docker-compose.yml    Reproducible local execution
requirements.runtime.txt  Runtime Python dependencies
test_ground_truth.py  Ground-truth parser/evaluation validation script
```

## MAIN WORKFLOWS

### Inspect A Dataset

Use the dataset panel to select a pair. The app shows the bivariate data, the threshold variable, the threshold value, and the two resulting regimes.

### Run Data-Driven Causal Methods

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

## SAMPLING POLICY

For batch LLM experiments:

- Repeated calls use deterministic seeds starting at 42.
- Raw-data configurations pass the full pair when it is below the configured row cap.
- Larger raw-data pairs are sampled directly from the full pair for each repeated run.
- Segmented-regime configurations split by the metadata threshold first.
- Larger regimes are sampled after segmentation.
- Repeated outputs are aggregated by majority vote per regime before scoring.

The row preview used by the UI is separate from the batch-experiment sampling path.

## DATA AND METADATA

Dataset files live in:

```text
DATA/custom_pairs/
```

Ground-truth threshold and direction metadata lives in
`DATA/pairmeta_with_ground_truth.txt`. Each metadata row specifies the dataset
identifier, the cause/effect column indices, a dataset weight, the threshold
variable, the threshold value, and the causal direction in the low and high
regime.

Detailed per-dataset descriptions (data sources, the domain literature used to
fix each regime threshold, and the ground-truth causal directions) are given
in the technical supplement, in the section "Dataset Descriptions".


## REPRODUCING THE VALIDATION CHECK

After building the Docker image, run:

```bash
docker compose exec causal-app python test_ground_truth.py
```

Expected outcome: the script completes without ground-truth lookup errors.


## ENVIRONMENT VARIABLES

| Variable | Required | Description |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | Required for LLM runs | OpenRouter API key |
| `OPENROUTER_HTTP_REFERER` | No | Optional HTTP referer header |
| `OPENROUTER_X_TITLE` | No | Optional title header |


## EXTERNAL METHODS

This app uses or wraps the following causal-discovery methods for comparison.
Raw third-party benchmark data is not redistributed and remains subject to the
original terms of the sources cited in the paper.

- LOCI, Location-Scale Noise Models: <https://github.com/AlexImmer/loci>
- ROCHE, robust causal discovery with Student's t-distribution: <https://github.com/quangdzuytran/ROCHE>
- LCUBE, MDL-based causal discovery with cubic splines: <https://github.com/LCube-Alg/LCube>
- LiNGAM: <https://github.com/cdt15/lingam>

## NOTES FOR REVIEWERS

This package is intended for anonymous academic review. It contains processed
datasets and runnable code, but no private API keys. LLM experiments require
the reviewer to provide their own OpenRouter key.
