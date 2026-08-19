"""
SalesNeuron — LLM Client
Primary: Gemini 2.5 Flash (free, 1500 req/day, 1M context)
Fallback: Groq Llama 3.3 70B (free, fast, 100K TPD)

Usage:
    from core.llm import llm
    response = await llm.generate("your prompt here")
    structured = await llm.generate_structured("prompt", schema_dict)
"""

import os
import json
import asyncio
import logging
from typing import Optional
from google import genai
from google.genai import types as genai_types
from groq import Groq

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self):
        gemini_key = os.getenv("GEMINI_API_KEY", "")
        groq_key = os.getenv("GROQ_API_KEY", "")

        self._gemini_client = None
        self._groq_client = None

        if gemini_key and gemini_key != "your_gemini_api_key_here":
            self._gemini_client = genai.Client(api_key=gemini_key)
            logger.info("✅ Gemini 3.5 Flash ready (primary LLM)")
        else:
            logger.warning("⚠️  No Gemini key — set GEMINI_API_KEY in .env")

        if groq_key:
            self._groq_client = Groq(api_key=groq_key)
            logger.info("✅ Groq Llama 3.3 70B ready (fallback LLM)")

        if not self._gemini_client and not self._groq_client:
            logger.warning(
                "⚠️  No LLM configured. Add GEMINI_API_KEY or GROQ_API_KEY to your .env file.\n"
                "  Gemini (free): https://aistudio.google.com\n"
                "  Groq   (free): https://console.groq.com"
            )

    async def generate(self, prompt: str, temperature: float = 0.3) -> str:
        """Generate a text response. Tries Gemini first, falls back to Groq."""
        if self._gemini_client:
            try:
                return await self._gemini_generate(prompt, temperature)
            except Exception as e:
                logger.warning(f"Gemini failed ({e}), trying Groq fallback...")

        if self._groq_client:
            return await self._groq_generate(prompt, temperature)

        raise RuntimeError(
            "No LLM configured. Add GEMINI_API_KEY or GROQ_API_KEY to your .env file.\n"
            "  Gemini (free): https://aistudio.google.com\n"
            "  Groq   (free): https://console.groq.com"
        )

    async def generate_structured(
        self,
        prompt: str,
        output_description: str,
        temperature: float = 0.1,
    ) -> dict:
        """
        Generate a JSON-structured response.
        The model is instructed to return ONLY valid JSON.
        """
        structured_prompt = (
            f"{prompt}\n\n"
            f"Return ONLY a valid JSON object matching this structure:\n"
            f"{output_description}\n\n"
            f"No markdown, no explanation, no code fences. Pure JSON only."
        )
        raw = await self.generate(structured_prompt, temperature=temperature)
        return self._parse_json(raw)

    async def _gemini_generate(self, prompt: str, temperature: float) -> str:
        config = genai_types.GenerateContentConfig(temperature=temperature)
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._gemini_client.models.generate_content(
                model="gemini-3.5-flash",
                contents=prompt,
                config=config,
            ),
        )
        return response.text.strip()

    async def _groq_generate(self, prompt: str, temperature: float) -> str:
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._groq_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=4096,
            ),
        )
        return response.choices[0].message.content.strip()

    @staticmethod
    def _parse_json(raw: str) -> dict:
        """Strip markdown fences and parse JSON safely."""
        text = raw.strip()
        # Remove ```json ... ``` or ``` ... ```
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object within the text
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(text[start:end])
            raise ValueError(f"Could not parse JSON from LLM response:\n{raw[:300]}")


# Singleton — import this everywhere
llm = LLMClient()