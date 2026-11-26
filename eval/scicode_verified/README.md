# SciCode-Verified v2 standalone runner

This directory provides an experimental standalone runner for SciCode-Verified v2. It does not implement the TOP platform adapter. It uses the separate upstream repository for task text, reference data, prompts, skipped-step reference code and scoring helpers. Do not commit credentials, evaluation results, virtual environments or the HDF5 file.

## Linux prerequisites

- Python 3.11 for the controller; `bubblewrap` (`bwrap`) and `taskset` must be installed and unprivileged user namespaces must work.
- Two Linux Python 3.11 environments with NumPy/SciPy/h5py/SymPy/Matplotlib. Tested library versions: 2024: numpy 1.26.4, scipy 1.13.1, h5py 3.11.0, sympy 1.13.3, matplotlib 3.9.2; 2025: numpy 2.2.6, scipy 1.15.3, h5py 3.13.0, sympy 1.14.0, matplotlib 3.10.3. Other dependencies may be needed for specific tasks. Controller needs upstream imports and dependencies; install the upstream runtime requirements.
- A separate checkout of `https://github.com/flyingwagner/scicode-verified.git` at commit `0b40bd4a3a8a7820cd7eac68f1dbee590bac69b4`, and the v2 `test_data_cleaned.h5` in its `scicode_verified/` directory. Follow upstream's README for download. The runner verifies manifest checksums and requires 64 problems and 287 scored steps.
- A sandbox-compatible interpreter installation: a venv interpreter which resolves under a broad system or home root may be rejected by the sandbox preflight. Do not weaken isolation to bypass this check.

## Run

Set the API key in your shell without recording it in a config file. `--base-url` is the OpenAI-compatible API root including `/v1` for `openai_chat`; for `anthropic`, use the service base URL or `/v1`. The runner does not currently implement OpenAI Responses.

```bash
export SCICODE_VERIFIED_API_KEY='set-interactively'
python eval/scicode_verified/run.py \
  --source /path/to/scicode-verified \
  --env-2024 /path/to/science2024/bin/python \
  --env-2025 /path/to/science2025/bin/python \
  --output /path/to/new-experiment \
  --model MODEL_NAME --base-url https://example.com/v1 \
  --only 5 --rounds 1 --workers 1 --grading-workers 1 --cpu-slots 1
```

`run.py` refuses an existing output directory and stores only the key's environment variable name. Remove `--only` for all 64 problems. The default is three independent rounds with background, 1800 seconds per scoring environment, and 2024 OR 2025 grading. `--help` lists the concurrency and limit options. `pipeline.py --experiment <existing-experiment.json> --resume-source <previous-outputs>` is a low-level recovery interface that validates frozen source, config and saved output; do not edit a completed experiment in place. A successful full run writes `outputs/final-summary.json`; an incomplete run must not be interpreted as a full score.

Sub is the number of passing scored steps divided by the number scored; Main is the fraction of problems with all scored steps passing. For three full rounds the denominators are 861 steps and 192 problem trials. Responses consisting solely of `!`, truncated responses and missing terminal events are treated as incomplete, not as valid code. Retrying a model answer is a separate attempt, not a new official pass@1 sample. The runner preserves model imports, which differs from the upstream extraction method and should be disclosed when comparing scores.

## Validation status

The original personal-workspace pipeline completed real model evaluations and was tested with the pinned data and two environments. This standalone packaging needs its own clean-environment CLI and model smoke validation. OpenAI Responses and TOP integration have not been implemented or validated. The staged SB3 adapter is not needed by this standalone runner and must not be treated as a TOP adapter.

## Example environment preparation

The following installs the directly checked scientific libraries; it is not a fully locked reproduction environment. Use a dedicated directory outside this Git checkout and inspect any additional dependencies needed by your selected tasks.

```bash
# Requires git, uv, curl, bubblewrap and util-linux/taskset.
mkdir -p "$HOME/scicode-verified-runtime"
cd "$HOME/scicode-verified-runtime"
git clone https://github.com/flyingwagner/scicode-verified.git upstream
cd upstream
git checkout --detach 0b40bd4a3a8a7820cd7eac68f1dbee590bac69b4
# Follow upstream README to download test_data_cleaned.h5 into scicode_verified/.
cd ..
uv python install 3.11
uv venv --python 3.11 science2024
uv venv --python 3.11 science2025
# From this SciCode checkout, set REPO to its absolute path.
uv pip install --python "$PWD/science2024/bin/python" -r "$REPO/eval/scicode_verified/requirements-2024.txt"
uv pip install --python "$PWD/science2025/bin/python" -r "$REPO/eval/scicode_verified/requirements-2025.txt"
```

Confirm the pinned upstream commit and the HDF5 checksum match `scicode_verified/manifest.json`. The controller verifies both data checksums before running. Actual availability of bubblewrap namespaces and image sandbox permissions must be tested in the intended execution environment.

To run the lightweight local tests from the SciCode checkout:

```bash
python -B -m unittest discover -s eval/scicode_verified -p test_pipeline.py -v
```
