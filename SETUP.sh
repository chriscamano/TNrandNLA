#!/usr/bin/env bash
set -euo pipefail

GREEN="$(printf '\033[0;32m')"
RESET="$(printf '\033[0m')"

log() {
  printf '%s[ %s ]%s %s\n' "${GREEN}" "$1" "${RESET}" "${2-}"
}

ENV_NAME="${1:-tnrnla-mkl}"
CREATED_ENV="0"
log "TNrSVD SETUP INFO" "Using conda env: ${ENV_NAME}"

if ! command -v conda >/dev/null 2>&1; then
  log "ERROR" "conda not found in PATH"
  exit 1
fi

export CPATH="${CPATH-}"

eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  log "TNrSVD SETUP INFO" "Creating conda env: ${ENV_NAME}"
  conda create -n "${ENV_NAME}" -c conda-forge -y python=3.12 pip
  CREATED_ENV="1"
fi

if [ "${CREATED_ENV}" = "1" ]; then
  log "TNrSVD SETUP INFO" "Activating newly created env: ${ENV_NAME}"
else
  log "TNrSVD SETUP INFO" "Activating env: ${ENV_NAME}"
fi
set +u
conda activate "${ENV_NAME}"
set -u
export CPATH="${CPATH-}"

log "TNrSVD SETUP INFO" "Upgrading pip"
python -m pip install --upgrade pip

log "TNrSVD SETUP INFO" "Installing base Python deps"
python -m pip install ipykernel pyyaml line_profiler tqdm

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OS_NAME="$(uname -s)"
IS_MAC="0"

if [ "${OS_NAME}" = "Darwin" ]; then
  log "TNrSVD SETUP INFO" "Detected macOS"
  read -r -p "Are you on a Mac and want to skip MKL + incrementalqr C++ build? (y/n) " ans
  case "${ans}" in
    y|Y|yes|YES) IS_MAC="1" ;;
    *) IS_MAC="0" ;;
  esac
fi

if [ "${IS_MAC}" = "1" ]; then
  log "TNrSVD SETUP INFO" "macOS mode enabled"
  log "TNrSVD SETUP INFO" "Installing NumPy/SciPy from standard indexes and skipping MKL headers and C++ extension build"
  python -m pip install numpy scipy
else
  log "TNrSVD SETUP INFO" "Linux or MKL mode enabled"
  log "TNrSVD SETUP INFO" "Installing NumPy/SciPy from MKL wheel index"
  python -m pip install numpy scipy --extra-index-url https://urob.github.io/numpy-mkl

  log "TNrSVD SETUP INFO" "Installing MKL headers"
  set +u
  conda install -c conda-forge -y mkl-devel
  set -u
  export CPATH="${CPATH-}"

  log "TNrSVD SETUP INFO" "Installing build toolchain (ninja/cmake/compilers/pybind11)"
  set +u
  conda install -c conda-forge -y ninja cmake compilers "pybind11>=2.12" "pybind11-global>=2.12"
  set -u
  export CPATH="${CPATH-}"
fi

log "TNrSVD SETUP INFO" "Installing plotting deps"
python -m pip install matplotlib plotly seaborn

log "TNrSVD SETUP INFO" "Registering Jupyter kernel"
python -m ipykernel install --user --name "${ENV_NAME}" --display-name "Python (${ENV_NAME})"

cd "${REPO_ROOT}"
log "TNrSVD SETUP INFO" "Installing repo editable"
python -m pip install -e .

if [ "${IS_MAC}" = "1" ]; then
  log "TNrSVD SETUP INFO" "Skipping incrementalqr C++ extension build on macOS"
else
  log "TNrSVD SETUP INFO" "Building incrementalqr C++ extension"
  cd "${REPO_ROOT}/src/tnrnla/linalg/incrementalqr"
  rm -rf build

  cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython3_EXECUTABLE="$(python -c 'import sys; print(sys.executable)')"

  cmake --build build -j
  cmake --install build
fi

log "DONE" "Done."
log "DONE" "Select kernel: Python (${ENV_NAME})"
