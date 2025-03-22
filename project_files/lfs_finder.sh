#!/bin/bash

# Set size threshold (e.g., 100MB = 104857600 bytes)
SIZE_THRESHOLD=104857600

echo "Searching for files that should be tracked with Git LFS..."

# Loop through all files recursively
find . -type f ! -path "./.git/*" | while read -r file; do
  # Get file size in bytes (macOS version)
  filesize=$(stat -f%z "$file")
  
  # Check if file size is an integer and greater than threshold
  if [[ "$filesize" =~ ^[0-9]+$ ]] && [ "$filesize" -ge "$SIZE_THRESHOLD" ]; then
    echo "[LARGE FILE] $file ($filesize bytes)"
  fi
done

# Optional: detect by file type (e.g., media, binary formats)
echo -e "\nAlso consider tracking these types:"
find . -type f ! -path "./.git/*" \( \
  -iname "*.mp4" -o \
  -iname "*.mov" -o \
  -iname "*.zip" -o \
  -iname "*.tar.gz" -o \
  -iname "*.bin" -o \
  -iname "*.iso" -o \
  -iname "*.h5" -o \
  -iname "*.tar" -o \
  -iname "*.jpg" -o \
  -iname "*.png" \
\) -print

echo -e "\nDone. Consider using 'git lfs track <pattern>' for these files."

