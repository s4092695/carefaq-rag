# gen/lc_llm.py
import os
from typing import Literal

LLM_BACKEND: Literal["ollama","openai"] = os.getenv("LC_LLM_BACKEND", "ollama")
LLM_MODEL = os.getenv("LC_LLM_MODEL", "llama3.1")  # e.g., "llama3.1", "qwen2.5", or OpenAI "gpt-4o-mini"

def make_llm():
    if LLM_BACKEND == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=LLM_MODEL, temperature=0)  # deterministic for eval
    else:
        # OpenAI path (set OPENAI_API_KEY)
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=LLM_MODEL, temperature=0)
