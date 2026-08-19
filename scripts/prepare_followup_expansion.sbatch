#!/usr/bin/env bash
#SBATCH --job-name=paper-followup-prep
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/followup_prep_%j.out
#SBATCH --error=logs/followup_prep_%j.err

set -euo pipefail

ROOT="${PAPER_MINING_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PAPER_MINING_PYTHON:-python}"
cp -a "${ROOT}/discovery/all_papers_cache.json" \
    "${ROOT}/discovery/all_papers_cache_320000.json"
"${PYTHON_BIN}" \
    "${ROOT}/scripts/prepare_followup_expansion.py"
