#!/bin/bash

set -e

echo "Running type checking..."

echo "🔍 Type checking with mypy..."
uv run mypy backend/ main.py

echo "✅ Type checking complete!"