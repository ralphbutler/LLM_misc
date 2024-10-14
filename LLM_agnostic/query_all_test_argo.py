import query_all
import os

text_messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Tell me a joke."},
    {"role": "assistant", "content": "Why did the scarecrow win an award?\nBecause he was outstanding in his field!"},
    {"role": "user", "content": "Great! Tell me another!"},
]

# https_proxy_orig = os.environ.get("https_proxy")
# os.environ["https_proxy"] = "socks5h://localhost:32000"
response = query_all.generate("argo/gpt-4o-mini", text_messages)
# os.environ["https_proxy"] = https_proxy_orig or ''
print(response)
