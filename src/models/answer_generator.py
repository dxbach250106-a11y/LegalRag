from __future__ import annotations

from core.config import AppConfig
from retrieval.schemas import RetrievalResult


class LLMAnswerGenerator:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.settings = config.section("generation")
        self.pipeline = None

    def load(self) -> "LLMAnswerGenerator":
        from transformers import pipeline
        model_path = self.config.root / self.settings["local_model_dir"]
        source = str(model_path if model_path.exists() else self.settings["model_name"])
        self.pipeline = pipeline("text-generation", model=source, tokenizer=source, device_map="auto")
        return self

    def generate(self, query: str, contexts: list[RetrievalResult]) -> str:
        if not self.settings.get("enabled", False):
            raise RuntimeError("generation.enabled is false")
        if self.pipeline is None:
            self.load()
        context_text = "\n\n".join(f"[{i}] {item.text}" for i, item in enumerate(contexts, start=1))
        prompt = f"Dựa duy nhất trên ngữ cảnh pháp luật sau, trả lời bằng tiếng Việt và nêu nguồn [n].\n\n{context_text}\n\nCâu hỏi: {query}\nTrả lời:"
        result = self.pipeline(prompt, max_new_tokens=int(self.settings["max_new_tokens"]), do_sample=float(self.settings.get("temperature", 0.0)) > 0, temperature=max(float(self.settings.get("temperature", 0.1)), 1e-5))
        return str(result[0]["generated_text"])[len(prompt):].strip()
