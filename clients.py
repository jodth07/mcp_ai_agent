import requests


class OllamaMistralClient:

    def __init__(self, server_url="http://localhost:11434", model="openhermes"):
        """This uses mac Ollama server runner on port 11434
        make sure model is compatible with available memory
        olloma pull openhermes
        """
        self.server_url = server_url
        self.model = model
        self.tool_schemas = None

    def generate(self, prompt):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        if self.tool_schemas:
            payload["tools"] = self.tool_schemas

        response = requests.post(f"{self.server_url}/api/generate", json=payload)

        if not response.ok:
            print(f"Server Error: {response.status_code} {response.text}")
            raise Exception("Ollama server error")

        data = response.json()
        return data["response"]
