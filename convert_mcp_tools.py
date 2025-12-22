import json

def convert_mcp_tools(input_file: str, output_file: str) -> None:
    """
    Convert mcp-tools.json to a simplified JSON file with only name and description.
    
    Args:
        input_file: Path to the input mcp-tools.json file
        output_file: Path to the output simplified JSON file
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract only name and description from each tool
    simplified_tools = [
        {
            "name": tool["name"],
            "description": tool["description"]
        }
        for tool in data["tools"]
    ]
    
    # Write the simplified data to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(simplified_tools, f, indent=2, ensure_ascii=False)
    
    print(f"Converted {len(simplified_tools)} tools to {output_file}")

if __name__ == "__main__":
    input_path = "data/mcp-tools.json"
    output_path = "data/mcp-tools-simplified.json"
    convert_mcp_tools(input_path, output_path)
