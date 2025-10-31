#!/bin/bash

# More precise script to remove only JSDoc comments
find frontend -name "*.ts" -not -path "*/node_modules/*" -not -path "*/.next/*" | while read file; do
  echo "Processing: $file"
  
  # Create a temporary file
  temp_file=$(mktemp)
  
  # Use awk to remove JSDoc comments more precisely
  awk '
  /^[[:space:]]*\/\*\*/ {
    # Start of JSDoc comment
    in_jsdoc = 1
    next
  }
  in_jsdoc && /^[[:space:]]*\*\/[[:space:]]*$/ {
    # End of JSDoc comment
    in_jsdoc = 0
    next
  }
  in_jsdoc {
    # Skip lines inside JSDoc comment
    next
  }
  {
    # Print all other lines
    print
  }
  ' "$file" > "$temp_file"
  
  # Replace original file with processed version
  mv "$temp_file" "$file"
done

echo "JSDoc comments removed from all TypeScript files"
