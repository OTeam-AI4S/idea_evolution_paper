#!/usr/bin/env bash
#SBATCH --job-name=paper-api-expand
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/api_expand_%j.out
#SBATCH --error=logs/api_expand_%j.err

set -euo pipefail

ROOT="${PAPER_MINING_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PAPER_MINING_PYTHON:-python}"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export PYTHONPATH="${ROOT}/runtime"
export PAPER_CANDIDATE_TARGET=320000

cd "${ROOT}"
"${PYTHON_BIN}" \
    scripts/expand_candidate_cache_api.py
