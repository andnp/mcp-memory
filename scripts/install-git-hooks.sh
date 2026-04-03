#!/bin/sh
set -eu

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

git config --local core.hooksPath .githooks
printf 'Configured core.hooksPath=.githooks for %s\n' "$repo_root"
