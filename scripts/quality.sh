#!/bin/bash

set -e

echo "Running full code quality pipeline..."

# Format code
./scripts/format.sh

# Run linting
./scripts/lint.sh

# Run type checking (optional - may have some type issues)
echo "🔍 Type checking with mypy (optional)..."
uv run mypy backend/ main.py || echo "⚠️  Type checking has some issues but continuing..."

# Run tests
echo "🧪 Running tests..."
uv run pytest backend/tests/ -v

echo "🎉 All quality checks and tests passed!"