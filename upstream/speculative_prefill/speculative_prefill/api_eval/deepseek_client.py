import asyncio
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import httpx


DEFAULT_ENV_FILES = (
    Path(".env"),
    Path.home() / ".deepseek.env",
    Path("/Users/jaso1024/Documents/compressioncompany/research/.env"),
)


def load_deepseek_api_key(env_file: str | None = None) -> str:
    if os.environ.get("DEEPSEEK_API_KEY"):
        return os.environ["DEEPSEEK_API_KEY"]

    paths = [Path(env_file)] if env_file else list(DEFAULT_ENV_FILES)
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("export "):
                stripped = stripped[len("export "):]
            if not stripped.startswith("DEEPSEEK_API_KEY="):
                continue
            value = stripped.split("=", 1)[1].strip().strip('"').strip("'")
            if value:
                return value
    raise RuntimeError("DEEPSEEK_API_KEY not found in environment or env files.")


@dataclass(frozen=True)
class DeepSeekResult:
    content: str
    latency_s: float
    usage: dict


class DeepSeekClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str = "deepseek-v4-flash",
        base_url: str = "https://api.deepseek.com",
        timeout: float = 120.0,
        retries: int = 3,
        concurrency: int = 8,
    ):
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.concurrency = concurrency
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        stop: str | None = None,
    ) -> DeepSeekResult:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if stop is not None:
            payload["stop"] = stop
        last_error = None
        for attempt in range(self.retries + 1):
            start = time.perf_counter()
            try:
                response = await self._client.post("/chat/completions", json=payload)
                if response.status_code >= 400:
                    raise RuntimeError(
                        f"{response.status_code} {response.reason_phrase}: "
                        f"{response.text[:1000]}"
                    )
                data = response.json()
                return DeepSeekResult(
                    content=data["choices"][0]["message"]["content"],
                    latency_s=time.perf_counter() - start,
                    usage=data.get("usage", {}),
                )
            except Exception as exc:
                last_error = exc
                if attempt >= self.retries:
                    break
                await asyncio.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"DeepSeek request failed: {last_error}")

    async def run_jobs(self, jobs: Iterable[tuple[int, list[dict[str, str]], int, str | None]]):
        semaphore = asyncio.Semaphore(self.concurrency)

        async def run_one(job):
            idx, messages, max_tokens, stop = job
            async with semaphore:
                try:
                    return idx, await self.chat(messages, max_tokens, stop=stop)
                except Exception as exc:
                    return idx, exc

        tasks = [asyncio.create_task(run_one(job)) for job in jobs]
        for task in asyncio.as_completed(tasks):
            yield await task
