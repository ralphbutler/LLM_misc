from typing import Any, List, Mapping, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
import requests
import json
import os
from pydantic import Field

from enum import Enum

class ModelType(Enum):
    GPT35 = 'gpt35'
    GPT4 = 'gpt4'

class ArgoLLM(LLM):

    model_type: ModelType = ModelType.GPT35
    url: str = "https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/chat/"
    temperature: Optional[float] = 0.8
    system: Optional[str]
    top_p: Optional[float]= 0.7
    user: str = os.getenv("USER")
    
    @property
    def _llm_type(self) -> str:
        return "ArgoLLM"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:

        headers = {
            "Content-Type": "application/json"
        }
        params = {
            **self._get_model_default_parameters,
            **kwargs,
            "prompt": [prompt],
            "stop": []
        }

        params_json = json.dumps(params);
        print(params_json)
        response = requests.post(self.url, headers=headers, data=params_json)

        if response.status_code == 200:
            parsed = json.loads(response.text)
            return parsed['response']
        else:
            raise Exception(f"Request failed with status code: {response.status_code} {response.text}")

    @property
    def _get_model_default_parameters(self):
        return {
            "user": self.user,
            "model": self.model,
            "system": "" if self.system is None else self.system,
            "temperature": self.temperature,
            "top_p":  self.top_p
        }

    @property
    def model(self):
        return self.model_type.value
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {}

    @property
    def _generations(self):
        return

class ArgoEmbeddingWrapper():
    url: str = "https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/embed/"
    user: str = os.getenv("USER")

    @property
    def _llm_type(self) -> str:
        return "ArgoLLM"

    def _call(
        self, 
        prompts: List[str], 
        run_manager: Optional[CallbackManagerForLLMRun] = None, 
        **kwargs: Any
    ) -> str:
        headers = { "Content-Type": "application/json" }
        params = { 
            "user": self.user, 
            "prompt": prompts
        }
        params_json = json.dumps(params)
        response = requests.post(self.url, headers=headers, data=params_json)
        if response.status_code == 200:
            parsed = json.loads(response.text)
            return parsed
        else:
            raise Exception(f"Request failed with status code: {response.status_code} {response.text}")

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {}

    @property
    def _generations(self):
        return

    def embed_documents(self, texts):
        return self.invoke(texts)

    def embed_query(self, query):
        return self.invoke(query)[0]
