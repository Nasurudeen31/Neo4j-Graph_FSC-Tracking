import requests

grounded_prompt = "Answer the user's question using the provided context."

payload = {
    "model": "llama3.2:1b",
    "prompt": grounded_prompt,
    "stream": False,
    "options": {
        "temperature": 0.1
    }
}

response = requests.post(
    "http://localhost:11434/api/generate",
    json=payload,
    timeout=120
)

generated_answer = response.json()["response"]