#!/bin/bash
# Install dependencies on the NVIDIA (NGC) container. Re-run after a pod restart.
# The container ships PyTorch 2.4 and NumPy 1.x, so transformers/NumPy are pinned to compatible versions,
# and the container's unreachable NGC package index is bypassed.
export PIP_CONFIG_FILE=/dev/null
pip install --index-url https://pypi.org/simple -r "$(dirname "$0")/requirements.txt"
