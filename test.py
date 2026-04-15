import ollama

response = ollama.chat(
    model="llama3",
    messages=[{"role": "user", "content": "Plan a trip to Paris"}]
)

print(response['message']['content'])