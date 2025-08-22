#!/bin/bash

set -e

echo "Running code formatting..."

echo "🔧 Formatting with black..."
uv run black .

echo "🔧 Sorting imports with isort..."
uv run isort .

echo "✅ Code formatting complete!"