"""Minimal Jev client over httpx.

Two backends, same request/response shape:
  typesafe   -> POST {TYPESAFE_BASE_URL}/v1/systemone           (TYPESAFE_API_KEY)
  openrouter -> POST https://openrouter.ai/api/alpha/decisions  (OPENROUTER_API_KEY)

The repo's own probing/scripts/jev_client.py imports a vendored `httpclient`
module that was never committed, so this is a standalone replacement rather
than a fix to their tree.
"""

from __future__ import annotations

import os
import time

import httpx

OPENROUTER_URL = "https://openrouter.ai/api/alpha/decisions"


class JevError(RuntimeError):
    pass


class JevClient:
    def __init__(self, backend=None, model=None, timeout=60.0, max_calls=50):
        self.backend = backend or os.environ.get("JEV_BACKEND", "typesafe")
        if self.backend == "typesafe":
            base = os.environ.get("TYPESAFE_BASE_URL", "https://api.typesafe.ai").rstrip("/")
            self.url = base + "/v1/systemone"
            key = os.environ.get("TYPESAFE_API_KEY")
            self.model = model or "jev-latest"
        elif self.backend == "openrouter":
            self.url = OPENROUTER_URL
            key = os.environ.get("OPENROUTER_API_KEY")
            self.model = model or "~typesafe/jev-latest"
        else:
            raise JevError("unknown backend %r (want typesafe or openrouter)" % self.backend)
        if not key:
            raise JevError("missing API key for backend %r" % self.backend)

        self._http = httpx.Client(
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
            timeout=timeout,
        )
        self.max_calls = max_calls
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.api_seconds = 0.0
        self.latencies = []
        self.served_model = None   # what the API says it served, vs what we asked for

    def ask(self, state, questions):
        if self.calls >= self.max_calls:
            raise JevError(
                "call budget exhausted (%d calls); raise --max-calls if that is intended"
                % self.max_calls
            )
        payload = {"state": state, "model": self.model, "questions": questions}
        t0 = time.monotonic()
        r = self._http.post(self.url, json=payload)
        dt = time.monotonic() - t0

        self.calls += 1
        self.api_seconds += dt
        self.latencies.append(dt)
        if r.status_code != 200:
            raise JevError("HTTP %d: %s" % (r.status_code, r.text[:500]))

        data = r.json()
        usage = data.get("usage") or {}
        self.input_tokens += usage.get("input_tokens", 0)
        self.output_tokens += usage.get("output_tokens", 0)
        if data.get("model"):
            self.served_model = data["model"]
        data["_latency_s"] = round(dt, 4)
        return data

    def stats(self):
        lat = sorted(self.latencies)
        return {
            "backend": self.backend,
            "model_requested": self.model,
            "model_served": self.served_model,
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "api_seconds": round(self.api_seconds, 3),
            "median_latency_s": round(lat[len(lat) // 2], 4) if lat else None,
        }

    def close(self):
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
