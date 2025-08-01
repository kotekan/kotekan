#!/bin/sh

rsync -Paz --exclude .git --exclude __pycache__ --exclude Manifest\*.toml --exclude cmake-build\* --exclude data --exclude '*~' --exclude '*.log' --exclude python/env --del ~/src/kotekan-base-changes cx66:src
