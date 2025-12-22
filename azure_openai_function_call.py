import os
import json
from openai import AzureOpenAI

# Initialize the Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2025-03-01-preview",  # Responses API requires newer API version
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# Define the functions (tools) that the model can call
tools = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g., San Francisco, CA"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit"
                }
            },
            "required": ["location"]
        }
    }
]


# Simulated function to handle the tool call
def get_weather(location: str, unit: str = "celsius") -> dict:
    """Simulated weather function - replace with actual API call"""
    return {
        "location": location,
        "temperature": 22 if unit == "celsius" else 72,
        "unit": unit,
        "condition": "sunny"
    }


def main():
    # Make the initial API call using Responses API
    response = client.responses.create(
        model="gpt-4",  # Your deployment name
        instructions="You are a helpful assistant.",
        input="What's the weather like in Seattle?",
        tools=tools
    )

    # Process the response output
    function_calls = []
    text_output = None

    for item in response.output:
        if item.type == "function_call":
            function_calls.append(item)
        elif item.type == "message":
            for content in item.content:
                if content.type == "output_text":
                    text_output = content.text

    # Check if the model wants to call functions
    if function_calls:
        tool_outputs = []

        for func_call in function_calls:
            function_name = func_call.name
            function_args = json.loads(func_call.arguments)

            # Execute the function
            if function_name == "get_weather":
                function_response = get_weather(**function_args)

            # Collect tool outputs
            tool_outputs.append({
                "type": "function_call_output",
                "call_id": func_call.call_id,
                "output": json.dumps(function_response)
            })

        # Get the final response with tool outputs
        final_response = client.responses.create(
            model="gpt-4",
            previous_response_id=response.id,
            input=tool_outputs
        )

        # Extract the final text output
        for item in final_response.output:
            if item.type == "message":
                for content in item.content:
                    if content.type == "output_text":
                        print(content.text)
                        return

    elif text_output:
        print(text_output)


if __name__ == "__main__":
    main()
