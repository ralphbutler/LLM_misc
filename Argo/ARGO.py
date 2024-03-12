#
# A wrapper class for the Argonne Argo LLM service
#

import os
import requests
import json

MODEL_GPT35 = "gpt35"
MODEL_GPT4 = "gpt4"

class ArgoWrapper:
    default_url = "https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/chat/"

    def __init__(self, 
                 url = None, 
                 model = MODEL_GPT35, 
                 system = "",
                 temperature = 0.8, 
                 top_p=0.7, 
                 user = os.getenv("USER"))-> None:
        self.url = url
        if self.url is None:
            self.url = ArgoWrapper.default_url
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.user = user
        self.system = ""

    def invoke(self, prompt: str):
        headers = {
            "Content-Type": "application/json"
        }
        data = {
                "user": self.user,
                "model": self.model,
                "system": self.system,
                "prompt": [prompt],
                "stop": [],
                "temperature": self.temperature,
                "top_p": self.top_p
        }
            
        data_json = json.dumps(data)    
        response = requests.post(self.url, headers=headers, data=data_json)

        if response.status_code == 200:
            parsed = json.loads(response.text)
            return parsed
        else:
            raise Exception(f"Request failed with status code: {response.status_code}")

class ArgoEmbeddingWrapper:
    default_url = "https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/embed/"

    def __init__(self, url = None, user = os.getenv("USER")) -> None:
        self.url = url if url else ArgoEmbeddingWrapper.default_url
        self.user = user
        #self.argo_embedding_wrapper = argo_embedding_wrapper

    def invoke(self, prompts: list):
        headers = { "Content-Type": "application/json" }
        data = {
            "user": self.user,
            "prompt": prompts
        }
        data_json = json.dumps(data)
        response = requests.post(self.url, headers=headers, data=data_json)
        if response.status_code == 200:
            parsed = json.loads(response.text)
            return parsed
        else:
            raise Exception(f"Request failed with status code: {response.status_code}")

    def embed_documents(self, texts):
        return self.invoke(texts)

    def embed_query(self, query):
        return self.invoke(query)[0]
