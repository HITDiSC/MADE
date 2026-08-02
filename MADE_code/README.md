# MADE — Model Autonomous Deployment Engine (Supplementary Code)

This is the anonymized source code accompanying the paper submission. MADE is a
dual-agent LLM system that autonomously deploys a machine-learning model from a
research artifact (a paper PDF or a GitHub repository) into a running,
contract-conforming FastAPI inference service.

> **Anonymized release.** All credentials, private endpoints, and
> author-identifying strings have been removed from `config/global.yaml` and the
> source (replaced with `YOUR_*` placeholders). The code is provided for review
> and reproducibility; it is not runnable until you supply your own API keys and
> a Docker host.

## Architecture

Two coordinating agents drive a five-phase deployment pipeline:

- **PhaseManager (PM)** — global supervisor. Owns the artifact-state belief and
  decides *whether* to intervene mid-phase via a consistency gate over four
  predicates (redundant / premature / stale / stall).
- **ExecutionMaster (EM)** — local executor. Selects and runs the per-phase
  tools; may accept / negotiate / reject a PM instruction based on its
  first-hand tool observations.

Pipeline phases:

```
repoingest  ->  weightresolve  ‖  dockersetup  ->  apiadaptation  ->  servicedelivery
```

A deployment is judged by **contract conformance** (the served API's response
carries the format-comparable fields the reference output defines), not by
prediction accuracy.

## Directory layout

```
main.py                     Entry point: orchestrates the 5 phases + PM/EM loop
config/global.yaml          All knobs: turn budgets, coordination settings, keys (redacted)
agent/                      The two agents + coordination
  phase_manager.py            PM: belief + gap-check
  excution_master.py          EM: per-phase tool loop
  coordination.py             Consistency gate (4 predicates), override, invalidation
  parallel_discriminator.py   Merges the parallel weightresolve ‖ dockersetup lane
  output_store.py             Per-role structured-output schemas
  *_prompt.json               Agent prompts
agenttool/                  One module per phase + shared tools
  RepoIngest.py               PDF -> github link -> clone -> task selection
  WeightResolve.py            Resolve + download model weights (HF / GDrive / Kaggle / Zenodo)
  DockerSetUp.py              Generate + build the container image
  APIAdaptation.py            Adapt inference code into load/preprocess/inference/postprocess
  ServiceDelivery.py          Serve, validate, finalize the API
  evaluation.py               Standalone inference-result evaluation flow (normalize -> judge -> score)
  task_selection_tool.py, tool.py, entry_find_tool.py, gpu_*.py, FileTracker.py, ...
backend/
  query.py                    OpenAI-compatible LLM client (text / json / code queries) + retry
  logger.py                   JSONL call/event logger (LLM calls, gate decisions, gate stats)
prompts/                    All tool prompts (JSON templates)
prefile/                    FastAPI service template scaffolded into each deployment
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Also required (not a pip package): **Docker Engine** with the **NVIDIA container
runtime** for GPU inference.

Then edit `config/global.yaml` → `backend:` and fill in:

- `gr_base_url` + `gr_api_key` — your OpenAI-compatible LLM endpoint (used with
  `--backend gr`), **or** `openai_api_key` (used with `--backend openai`).
- optionally `google_drive_api_key`, `kaggle_api_key`, `kaggle_username` — only
  needed to download weights hosted on those services.
- `download_setting.repo_root` / `weights_root` — where cloned repos and
  downloaded weights are stored.

## Running

Exactly one input source is required (`--pdf_path`, `--github_link`, or
`--resume_from`):

```bash
# From a paper PDF (full ingest: parse PDF -> extract GitHub link -> clone -> select task)
python main.py --pdf_path path/to/paper.pdf \
               --test_file_dir path/to/testcases/case1 \
               --backend gr

# From a GitHub repo directly (skips PDF parsing)
python main.py --github_link https://github.com/<owner>/<repo> \
               --test_file_dir path/to/testcases/case1 \
               --backend gr

# Resume from a checkpoint (skips repoingest / weightresolve / dockersetup and
# restarts at the post-parallel phase, typically apiadaptation)
python main.py --resume_from path/to/variable_store_after_parallel.json \
               --test_file_dir path/to/testcases/case1 \
               --backend gr
```

`--test_file_dir` points at a test case directory containing `input/`,
`output/`, and (optionally) `OUTPUT_FORMAT.md`; it drives both run naming and
the contract-conformance validation in `servicedelivery`.

`--resume_from` takes a `variable_store_after_parallel.json` checkpoint (written
after the parallel weightresolve ‖ dockersetup lane completes) and restores the
variable store + PM board so a run can restart at `apiadaptation` without
re-ingesting the repo, re-resolving weights, or rebuilding the image.

## Logging

`backend/logger.py` writes a JSONL log per run: every LLM call (role, tokens,
duration, optional logprobs), plus `gate_decision` and `gate_stats` events for
the coordination mechanism — enough to reconstruct cost and intervention
statistics offline.
