from typing import Any, List, Mapping, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
import requests
import json
from ARGO import ArgoWrapper, ArgoEmbeddingWrapper


# The ARGO_LLM class. Uses the _invoke_model helper function.
# It implements the _call function.


class ARGO_LLM(LLM):

    argo: ArgoWrapper

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            print(f"STOP={stop}")
            # raise ValueError("stop kwargs are not permitted.")

        response = self.argo.invoke(prompt)       
        print(f"ARGO Response: {response['response']}\nEND ARGO RESPONSE")
        return response['response']

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {}

    @property
    def _generations(self):
        return

class ARGO_EMBEDDING:
    argo: ArgoEmbeddingWrapper
    def __init__(self, argo_wrapper: ArgoEmbeddingWrapper):
        self.argo = argo_wrapper
    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call( self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any, ) -> str:
        if stop is not None:
            print(f"STOP={stop}")
        response = self.argo.invoke(prompt)
        print(f"ARGO Response: {response['embedding']}\nEND ARGO RESPONSE")
        return response['embedding']

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {}

    @property
    def _generations(self):
        return

    def embed_documents(self, texts):
        return self._call(texts)

    def embed_query(self, query: str):
        # Handle embedding of a single query string
        # Assuming 'query' is a single string
        return self._call(query)[0]
