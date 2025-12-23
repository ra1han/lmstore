"""
Script to extract name and description from mcp-tools.json
and create a cleaner tools.json file.
"""

import json
from pathlib import Path


def main():
    # Define paths
    data_dir = Path(__file__).parent.parent / "data"
    input_file = data_dir / "raw_mcp_tools.json"
    output_file = data_dir / "tools.json"

    # Load the mcp-tools.json file
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract only name and description from each tool
    cleaned_tools = [
        {"name": tool["name"], "description": tool["description"]}
        for tool in data["tools"]
    ]

    # Save the cleaned data to tools.json
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cleaned_tools, f, indent=2, ensure_ascii=False)

    print(f"Successfully created {output_file}")
    print(f"Extracted {len(cleaned_tools)} tools")


if __name__ == "__main__":
    main()
