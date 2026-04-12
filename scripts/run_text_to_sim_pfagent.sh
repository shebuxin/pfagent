#!/usr/bin/env bash
set -euo pipefail

CONDA_HOME="/home/bshe/miniconda3"
ENV_NAME="pfagent"
ENV_PREFIX="${CONDA_HOME}/envs/${ENV_NAME}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source "${CONDA_HOME}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

export CONDA_ENV_PATH="${ENV_PREFIX}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-pfagent}"
mkdir -p "${MPLCONFIGDIR}"

cd "${REPO_ROOT}/text-to-sim"
exec streamlit run main.py
