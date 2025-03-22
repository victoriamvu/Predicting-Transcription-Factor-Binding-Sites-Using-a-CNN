#!/bin/bash

if [ -z "$1" ]; then
  echo "❌ Error: Please provide a prompt."
  echo "Usage: ./claude_context.sh \"Summarize what this project does and how it’s organized.\""
  exit 1
fi

# Find files (adjust extensions or exclusions as needed)
files=$(find . -type f \( -name "*.py" -o -name "*.sh" -o -name "*.md" -o -name "*.txt" -o -name "*.yml" -o -name "*.yaml" \) ! -path "*/.*/*" ! -size +100k)

# Create a temporary file for context
tempfile=$(mktemp /tmp/claude_input.XXXXXX)

for file in $files; do
  echo "### FILE: $file" >> "$tempfile"
  head -n 200 "$file" >> "$tempfile"
  echo -e "\n\n" >> "$tempfile"
done

# Combine your prompt and the file contents, then pipe into gpt-cli
{
  echo "$1"
  echo ""
  cat "$tempfile"
} | gpt general --prompt -

rm "$tempfile"

