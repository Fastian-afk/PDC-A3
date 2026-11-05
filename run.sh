#!/usr/bin/env bash
# Imaad Fazal | 23i-0656 | Section C | Assignment 3
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$project_dir/tools/cluster_run.py" "$@"
