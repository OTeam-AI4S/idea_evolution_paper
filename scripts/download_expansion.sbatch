#!/usr/bin/env bash
#SBATCH --job-name=paper-v7-more
#SBATCH --partition=gpu
#SBATCH --array=0-7%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/expansion_%A_%a.out
#SBATCH --error=logs/expansion_%A_%a.err
#SBATCH --requeue

set -euo pipefail

ROOT="${PAPER_MINING_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PAPER_MINING_PYTHON:-python}"
TASK_INDEX=${SLURM_ARRAY_TASK_ID}
OUTPUT_SHARD=$((TASK_INDEX + 8))
CACHE="${PAPER_MINING_SOURCE_CACHE:-${ROOT}/discovery/all_papers_cache.json}"

# Compute nodes can reach the open scholarly sources directly. The desktop
# session's loopback proxy is local to the login node and must not be inherited.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

export PYTHONPATH="${ROOT}/runtime"
export PAPER_MINING_OUTPUT_DIR="${ROOT}/shards/shard_${OUTPUT_SHARD}"
export PAPER_MINING_REFERENCE_METADATA_PATHS="${CACHE}"
export PAPER_MINING_DISABLE_GATEWAY=1

cd "${ROOT}/runtime"
"${PYTHON_BIN}" -m paper_mining.bulk_download \
    --download-only \
    --source-cache "${CACHE}" \
    --limit 20000 \
    --shard-count 8 \
    --shard-index "${TASK_INDEX}"
