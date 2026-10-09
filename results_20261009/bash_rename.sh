#!/bin/bash

TARGET_DIR="./decisions/"

OLD_PREFIXES=(
    "cyankiwi_Qwen3.5-9B-AWQ-4bit-local"
    "cyankiwi_Qwen3.5-9B-AWQ-4bit-cyankiwi_Qwen3.8-27B-AWQ-INT4__"
)

NEW_PREFIXES=(
    "cyankiwi_Qwen3.5-9B-AWQ-4bit"
    "cyankiwi_Qwen3.5-9B-AWQ-4bit-"
)

cd "$TARGET_DIR" || exit 1

for i in "${!OLD_PREFIXES[@]}"; do
    OLD_PREFIX="${OLD_PREFIXES[$i]}"
    NEW_PREFIX="${NEW_PREFIXES[$i]}"

    for file in "${OLD_PREFIX}"*; do
        [ -e "$file" ] || continue

        new_name="${NEW_PREFIX}${file#$OLD_PREFIX}"

        # echo "Renaming: '$file' -> '$new_name'"
	# echo "$new_name"
        # Dry run
        mv -n -- "$file" "$new_name"
    done
done

echo "Done!"
