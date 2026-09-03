from __future__ import annotations

import ast
import hashlib
import json
import os
from typing import Any, Dict, Iterable, List, Optional

try:  # pragma: no cover - optional runtime dependency
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
except Exception:  # pragma: no cover
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    BitsAndBytesConfig = None


class LLMWrapper:
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-13b-chat-hf",
        load_4bit: bool = True,
        max_seq_len: int = 2048,
        device: str = "cuda",
        use_mock: Optional[bool] = None,
    ):
        self.device = device
        self.max_seq_len = max_seq_len
        self.model_name = model_name
        env_mock = os.environ.get("TKL_XR_USE_MOCK_LLM", "0") == "1"
        self.use_mock = use_mock if use_mock is not None else env_mock
        dependencies_available = (
            torch is not None
            and AutoTokenizer is not None
            and AutoModelForCausalLM is not None
        )
        if use_mock is False and not dependencies_available:
            raise RuntimeError(
                "Llama inference was requested, but torch/transformers are not "
                "installed. Install requirements.txt and requirements-llm.txt, "
                "or pass --use_mock_llm for a deterministic smoke test."
            )
        self._mock_mode = bool(self.use_mock or not dependencies_available)

        self.tokenizer = None
        self.model = None
        self.bnb_config = None

        if self._mock_mode:
            return

        if load_4bit and BitsAndBytesConfig is not None:
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=self.bnb_config,
            device_map="auto" if device == "cuda" and torch is not None else None,
            torch_dtype=torch.bfloat16 if torch is not None else None,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def _mock_score(self, prompt: str) -> float:
        digest = hashlib.md5(prompt.encode("utf-8")).hexdigest()
        return (int(digest[:8], 16) % 1000) / 1000.0

    def _mock_generate(self, prompt: str) -> str:
        score = self._mock_score(prompt)
        return f"Mock response. Confidence={score:.3f}."

    def generate(self, prompt: str, temperature: float = 0.1, max_new_tokens: int = 512) -> str:
        if self._mock_mode:
            return self._mock_generate(prompt)

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_seq_len)
        inputs = inputs.to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def score(self, prompt: str) -> float:
        if self._mock_mode:
            return self._mock_score(prompt)

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_seq_len)
        inputs = inputs.to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        token_prob = torch.max(probs).item()
        return float(token_prob)

    def generate_with_score(self, prompt: str) -> tuple[str, float]:
        response = self.generate(prompt)
        return response, self.score(prompt + "\n" + response)

    def score_candidates(self, prompt: str, candidates: Iterable[str]) -> Dict[str, float]:
        candidate_scores: Dict[str, float] = {}
        for idx, candidate in enumerate(candidates):
            candidate_scores[str(candidate)] = self.score(f"{prompt}\nCandidate: {candidate}\nIndex: {idx}")
        return candidate_scores

    def parse_json_like(self, text: str) -> Dict[str, float]:
        text = text.strip()
        if not text:
            return {}
        try:
            return json.loads(text)
        except Exception:
            pass
        try:
            value = ast.literal_eval(text)
            if isinstance(value, dict):
                return {str(k): float(v) for k, v in value.items()}
        except Exception:
            pass
        return {}
