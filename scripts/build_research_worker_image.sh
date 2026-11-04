#!/bin/sh
set -eu
repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
context=$(mktemp -d)
trap 'rm -rf "$context"' EXIT
cp "$repo_root/Dockerfile.research-worker" "$context/"
cp "$repo_root/research-worker-requirements.txt" "$context/"
docker build --pull=false --file "$context/Dockerfile.research-worker" --tag multimodal-research-worker:2.7.0-cpu "$context"
