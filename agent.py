# agent.py

import json
import random
import requests

class BaseAgent:
    def __init__(self, model_client):
        """
        model_client: An object with a .generate(prompt: str) -> str method.
        """
        self.model_client = model_client
        self.memory = []  # Simple list for now

    def receive(self, user_message):
        """Handles incoming user messages."""
        conversation = self.format_conversation(user_message)
        model_output = self.model_client.generate(conversation)

        parsed_output = self.try_parse_json(model_output)

        if parsed_output and "tool_call" in parsed_output:
            # It's asking for a tool
            tool_result = self.call_tool(parsed_output["tool_call"], parsed_output.get("parameters", {}))
            response = f"(Tool result) {tool_result}"
        else:
            # It's a regular reply
            response = model_output

        self.memory.append({"user": user_message, "agent": response})
        return response

    def format_conversation(self, user_message):
        """Formats the conversation history + user message into a prompt."""
        history = "\n".join(
            f"User: {item['user']}\nAgent: {item['agent']}" for item in self.memory
        )
        return f"{history}\nUser: {user_message}\nAgent:"

    def try_parse_json(self, text):
        """Tries to parse text as JSON. Returns dict or None."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def call_tool(self, tool_name, parameters):
        """Stub to simulate tool calling."""
        if tool_name == "search_weather":
            city = parameters.get("city", "Unknown")
            # You would normally call an API here.
            return f"The weather in {city} is sunny and 85°F."
        else:
            return f"Unknown tool: {tool_name}"

class LocalMixtralClient:
    def __init__(self, server_url="http://localhost:5000"):
        self.server_url = server_url

    def generate(self, prompt):
        payload = {
            "prompt": prompt,
            "temperature": 0.3,
            "max_new_tokens": 500,
            "stop": None
        }
        response = requests.post(f"{self.server_url}/api/v1/generate", json=payload)
        data = response.json()

        return data["results"][0]["text"]

# Mock LLM client for now
class MockModelClient:
    def generate(self, prompt):
        """Fake generation: randomly decide to call a tool or answer."""
        if "weather" in prompt.lower():
            # Simulate tool call
            return json.dumps({
                "tool_call": "search_weather",
                "parameters": {"city": "Miami"}
            })
        else:
            responses = [
                "Sure! Here is what I found.",
                "Of course, let me help you with that.",
                "Here is the information you requested."
            ]
            return random.choice(responses)


if __name__ == "__main__":
    agent = BaseAgent(model_client=LocalMixtralClient())

    print("Agent is ready! Type your message:")
    while True:
        user_input = input("> ")
        reply = agent.receive(user_input)
        print(f"Agent: {reply}")
