import requests

# Build client so connections can be re-used across
# requests. This only needs to happen once per session.
client = requests.Session()

resp = client.post(
    "https://model-7qkkm18q.api.baseten.co/environments/production/predict",
    headers={"Authorization": "Api-Key t9mgrN9L.zFKfbzFVFb7zpx4lHNYSjUETYlisPTlQ"},
    json={'model': 'decagon-ai/select-playbook-prod', 'stream': True, 'messages': [{'role': 'user', 'content': 'Tell me everything you know about optimized inference.'}], 'max_tokens': 512, 'temperature': 0.6},
)

print(resp.json())