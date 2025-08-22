#!/bin/bash

set -e

echo "Running code quality checks..."

echo "🔍 Checking with flake8..."
uv run flake8 backend/ main.py

echo "🔍 Checking formatting with black..."
uv run black --check .

echo "🔍 Checking import sorting with isort..."
uv run isort --check-only .

echo "✅ All code quality checks passed!"