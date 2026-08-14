import os
from dataclasses import dataclass
from typing import Tuple
import logging
import time
import openai
from openai import OpenAI
import anthropic
from anthropic import Anthropic


class EmptyLLMResponseError(RuntimeError):
    """Raised when the LLM API returns a successful response with no text."""


class BaseLLM:

    def chat(self, messages, max_tokens=None) -> Tuple[str, dict]:
        raise NotImplementedError("Implement this method in the subclass.")

    def __call__(self, messages, max_tokens=None) -> Tuple[str, dict]:
        raise NotImplementedError("Implement this method in the subclass.")


class OpenAILLM(BaseLLM):
    api_retries: int = 3 # Number of retries to the LLM API
    min_retry_wait_s: float = 20.0

    def __init__(
        self,
        model_name,
        base_url=None,
        api_key=None,
        temperature=1.0,
        max_tokens=1024,
        log_probs=False,
        extra_body=None,
    ):
        
        base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        api_key = api_key or os.environ.get("OPENAI_API_KEY")

        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.log_probs = log_probs
        self.extra_body = extra_body

    def chat(self, messages, max_tokens=None):
        max_tokens = self.max_tokens if max_tokens is None else max_tokens
        request_kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_completion_tokens": max_tokens,
            "logprobs": self.log_probs,
        }
        if self.extra_body is not None:
            request_kwargs["extra_body"] = self.extra_body
        completion = self.client.chat.completions.create(**request_kwargs)
        response = completion.choices[0].message.content
        if response is None or not response.strip():
            raise EmptyLLMResponseError(
                f"Model {self.model_name} returned empty message content."
            )
        usage = completion.usage.to_dict()
        return response, {"usage": usage}

    def __call__(self, messages, max_tokens=None) -> Tuple[str, dict]:
        last_exc = None
        for retry in range(self.api_retries):
            try:
                return self.chat(messages, max_tokens=max_tokens)

            except openai.RateLimitError as e:
                last_exc = e
                logging.warning(
                    f"Rate limited ({retry + 1}/{self.api_retries}). "
                    f"Sleeping {self.min_retry_wait_s}s. [err]: {e}"
                )
                time.sleep(self.min_retry_wait_s)

            except openai.APIStatusError as e:
                # Retry transient server errors only; surface client errors immediately.
                if e.status_code and e.status_code >= 500:
                    last_exc = e
                    logging.warning(
                        f"Server error {e.status_code} ({retry + 1}/{self.api_retries}). "
                        f"Sleeping {self.min_retry_wait_s}s. [err]: {e}"
                    )
                    time.sleep(self.min_retry_wait_s)
                else:
                    raise

            except openai.APIConnectionError as e:
                last_exc = e
                logging.warning(
                    f"Connection error ({retry + 1}/{self.api_retries}). "
                    f"Sleeping {self.min_retry_wait_s}s. [err]: {e}"
                )
                time.sleep(self.min_retry_wait_s)

            except EmptyLLMResponseError as e:
                last_exc = e
                logging.warning(
                    f"Empty LLM response ({retry + 1}/{self.api_retries}). "
                    f"Sleeping {self.min_retry_wait_s}s. [err]: {e}"
                )
                time.sleep(self.min_retry_wait_s)

        if isinstance(last_exc, EmptyLLMResponseError):
            raise EmptyLLMResponseError(
                f"API call returned empty LLM response after {self.api_retries} retries."
            ) from last_exc

        raise RuntimeError(
            f"API call failed after {self.api_retries} retries."
        ) from last_exc


class OpenAIResponsesLLM(OpenAILLM):
    """OpenAI client using the Responses API instead of Chat Completions.

    This class intentionally leaves ``OpenAILLM`` unchanged. It reuses that
    class's client initialization and retry behavior while overriding only the
    endpoint-specific request and response extraction.
    """

    @staticmethod
    def _normalize_messages(messages):
        """Flatten legacy all-text content blocks for the Responses API.

        The prompt builders use Chat Completions-style ``text`` content blocks,
        which are not valid Responses API input content types. Responses accepts
        plain string content, so flatten lists made entirely of those legacy
        blocks while leaving strings and Responses-native/multimodal content
        unchanged. Copy each message so the caller's prompt is not mutated.
        """
        normalized_messages = []
        for message in messages:
            normalized_message = dict(message)
            content = normalized_message.get("content")
            if (
                isinstance(content, list)
                and content
                and all(
                    isinstance(part, dict)
                    and part.get("type") == "text"
                    and isinstance(part.get("text"), str)
                    for part in content
                )
            ):
                normalized_message["content"] = "\n".join(
                    part["text"] for part in content
                )
            normalized_messages.append(normalized_message)
        return normalized_messages

    def chat(self, messages, max_tokens=None):
        max_tokens = self.max_tokens if max_tokens is None else max_tokens
        request_kwargs = {
            "model": self.model_name,
            "input": self._normalize_messages(messages),
            "max_output_tokens": max_tokens,
        }
        if self.extra_body is not None:
            request_kwargs["extra_body"] = self.extra_body
        completion = self.client.responses.create(**request_kwargs)
        response = completion.output_text
        if response is None or not response.strip():
            raise EmptyLLMResponseError(
                f"Model {self.model_name} returned empty response output text."
            )
        usage = completion.usage.to_dict() if completion.usage is not None else {}
        return response, {"usage": usage}


class AnthropicLLM(BaseLLM):
    api_retries: int = 3 # Number of retries to the LLM API
    min_retry_wait_s: float = 20.0

    def __init__(
        self,
        model_name,
        base_url=None,
        api_key=None,
        max_tokens=2048,
        use_prompt_caching=True,
    ):
        base_url = base_url or os.environ.get("ANTHROPIC_BASE_URL")
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

        assert base_url is not None and base_url
        assert api_key is not None and api_key

        self.client = Anthropic(
            base_url=base_url,
            api_key=api_key
        )

        self.model_name = model_name
        self.max_tokens = max_tokens
        self.use_prompt_caching = use_prompt_caching

    @staticmethod
    def _with_cache_control(content):
        """Return a content-block list with cache_control on the last block.

        Accepts either a plain string or an existing list of content blocks
        (e.g. tool results / images) and marks the last block as an ephemeral
        cache breakpoint.
        """
        if isinstance(content, str):
            return [
                {
                    "type": "text",
                    "text": content,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        # Already a list of blocks: copy and annotate the last one so we don't
        # mutate the caller's message objects.
        blocks = [dict(block) for block in content]
        if blocks:
            blocks[-1] = {**blocks[-1], "cache_control": {"type": "ephemeral"}}
        return blocks

    def chat(self, messages, max_tokens=None):
        max_tokens = self.max_tokens if max_tokens is None else max_tokens
        system_message = messages[0]["content"]
        messages = messages[1:]

        if self.use_prompt_caching:
            # Breakpoint 1: cache the (stable) system prompt — this is the
            # largest reusable prefix, so it delivers most of the savings.
            system_message = self._with_cache_control(system_message)
            # Breakpoint 2: cache the conversation prefix so multi-turn /
            # agentic loops reuse prior turns. Mark the final message; copy it
            # first so the caller's list is left untouched.
            if messages:
                messages = list(messages)
                last = dict(messages[-1])
                last["content"] = self._with_cache_control(last["content"])
                messages[-1] = last

        completion = self.client.messages.create(
            model=self.model_name,
            system=system_message,
            messages=messages,
            max_tokens=max_tokens,
        )
        response = completion.content[0].text
        if response is None or not response.strip():
            raise EmptyLLMResponseError(
                f"Model {self.model_name} returned empty message content."
            )
        usage = completion.usage.to_dict()
        return response, {"usage": usage}

    def __call__(self, messages, max_tokens=None) -> Tuple[str, dict]:
        last_exc = None
        for retry in range(self.api_retries):
            try:
                return self.chat(messages, max_tokens=max_tokens)

            except anthropic.RateLimitError as e:
                last_exc = e
                logging.warning(
                    f"Rate limited ({retry + 1}/{self.api_retries}). "
                    f"Sleeping {self.min_retry_wait_s}s. [err]: {e}"
                )
                time.sleep(self.min_retry_wait_s)

            except anthropic.APIStatusError as e:
                # Retry transient server errors only; surface client errors immediately.
                if e.status_code and e.status_code >= 500:
                    last_exc = e
                    logging.warning(
                        f"Server error {e.status_code} ({retry + 1}/{self.api_retries}). "
                        f"Sleeping {self.min_retry_wait_s}s. [err]: {e}"
                    )
                    time.sleep(self.min_retry_wait_s)
                else:
                    raise

            except anthropic.APIConnectionError as e:
                last_exc = e
                logging.warning(
                    f"Connection error ({retry + 1}/{self.api_retries}). "
                    f"Sleeping {self.min_retry_wait_s}s. [err]: {e}"
                )
                time.sleep(self.min_retry_wait_s)

            except EmptyLLMResponseError as e:
                last_exc = e
                logging.warning(
                    f"Empty LLM response ({retry + 1}/{self.api_retries}). "
                    f"Sleeping {self.min_retry_wait_s}s. [err]: {e}"
                )
                time.sleep(self.min_retry_wait_s)

        if isinstance(last_exc, EmptyLLMResponseError):
            raise EmptyLLMResponseError(
                f"API call returned empty LLM response after {self.api_retries} retries."
            ) from last_exc

        raise RuntimeError(
            f"API call failed after {self.api_retries} retries."
        ) from last_exc
