# Contributing to ROCm-LS Examples

Thank you for your interest in contributing! This repository hosts examples 
and use-cases for products in the ROCm-LS organization (e.g., MONAI and 
hipCIM). Our goals are clarity, reproducibility, and ROCm-readiness 
across all examples.

This guide covers how to propose changes, coding standards, documentation 
expectations, and review processes.

## Code of Conduct

This project adheres to AMD's Code of Conduct. Be respectful, inclusive, 
and constructive. Report unacceptable behavior through AMD channels or 
via repository maintainers.

## Repository Structure

- Top-level folder for shared materials.
- Each scenario should be self-contained where practical:
   - Scripts (`train.py`, `infer.py`, etc.)
   - README.md describing usage and data requirements
   - Optional configs (YAML), `requirements.txt`
   - Small smoke tests if applicable

## Getting Started

- Fork the repository and create a topic branch
- Set up a Python environment and install dependencies
- Follow the coding standards and documentation requirements below
- Submit a pull request (PR) with a clear title and description

## Licensing and Notices

All new or modified source files must include SPDX and AMD copyright.
Example header for Python files:
```
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc.
```

For non-code files (e.g., documentation), include the SPDX license at the bottom if appropriate.

## Contribution Types

We welcome:
- New use-case examples (scripts + docs)
- Refinements to existing examples (API flags, logging, device handling)
- Documentation improvements (READMEs, comments, diagrams)
- Small, targeted utilities that reduce duplication across examples

Helpful guidelines:
- Default scripts should focus on demonstrative use-cases
- Include device detection and report HIP/CUDA versions where helpful

Please avoid:
- Large frameworks or complex shared libraries in this repo
- Proprietary or non-redistributable datasets or assets

## Branching, Commits, and Pull Requests

- Use feature branches: feature/<short-description>, fix/<short-description>, docs/<short-description>
- Write clear, imperative commit messages:
   - "Add unified MONAI inference CLI for spleen"
   - "Fix device selection and add ROCm version logging"
- Keep changes focused; avoid mixing unrelated updates
- Include a PR checklist (see below)

### PR checklist:

- [] Follows licensing header requirements
- [] Uses consistent CLI flags and logging
- [] Includes or updates README.md for the scenario
- [] Includes minimal smoke test or validation notes
- [] ROCm device selection handled (GPU preferred, CPU fallback)
- [] No hardcoded paths; uses CLI arguments
- [] Dependencies documented/pinned where necessary

## Coding Standards

- Language: Python 3.10+ (unless otherwise specified)
- Style: PEP8 + type hints for new functions where practical
- Naming:
   - Directories: hyphenated (e.g., `spleen-ct-segmentation`)
   - Files: snake_case (`train.py`, `infer.py`, `cli.py`)
- CLI consistency across scripts:
   - `--input`, `--input-dir`, `--output-dir`
   - `--bundle-dir` or `--model-path` when relevant
   - `--batch-size`, `--num-workers`
   - `--epochs` (training scripts)
   - `--seed`, `--verbose`/`-v`
   - `--device gpu|cpu` (where applicable)
- Logging: use logging or structured prints; avoid noisy output by default

## Documentation Standards

Each scenario must include README.md with:
- What the example demonstrates
- Prerequisites and environment setup (ROCm notes; PyTorch install channel)
- Data requirements and expected folder layout
- How to run (commands for typical use)
- Expected outputs (brief summary)
- Notes on GPU/CPU behavior, mixed precision, and performance tuning

### Data

- Describe dataset and paths
- Expected structure

### Commands

```
python train.py --data-root <path> --epochs 10
python infer.py --input <image> --save-output
```

### Outputs

- Brief description of expected artifacts/logs

### Notes

- GPU vs. CPU, mixed precision, optional flags

## Dependencies

- Pin versions where ROCm compatibility is critical (e.g., PyTorch build)
- Prefer per-scenario requirements.txt when dependencies vary significantly; otherwise use product-level requirements
- Avoid adding heavy dependencies unless essential for the example

## Data Handling
- Do not commit large datasets or model checkpoints
- Provide download scripts or instructions (e.g., MONAI apps download_url)
- Use relative paths via CLI args; avoid hardcoded absolute paths
- If using sample data, ensure redistribution is permitted and documented

## Reproducibility
- Provide seed options and set determinism where reasonable
- Document environment variables or CUDA/HIP flags if applicable
- Consider adding a lightweight smoke test:
   - Verifies environment, loads a small sample, and executes a short path through code.

## Testing

- Include minimal smoke tests or instructions to validate the environment (e.g., run with a single patch/tile)
- Optional: add unit tests for small utility functions
- CI integration is welcome via GitHub Actions (linting, minimal run checks), but keep jobs lightweight

## Review and Approval Process

- A maintainer will review PRs for:
   - Licensing headers
   - ROCm readiness (device selection, GPU/CPU fallback)
   - Documentation quality and clarity
   - Consistent naming and CLI patterns
- Reviewers may request changes; please respond constructively and promptly.

## Security and Privacy

- Do not include sensitive information (credentials, internal endpoints)
- Ensure any downloaded datasets are from trusted sources and documented
- Avoid writing files outside user-controlled output directories

## Contacts and Support

- Open an issue for questions or bugs
- Tag maintainers for reviews when PR is ready
- For ROCm build issues, refer to PyTorch ROCm docs and MONAI's installation guide

## Acknowledgements

Thanks to contributors across AMD and the open-source community for making 
ROCm-ready AI examples accessible and useful.

