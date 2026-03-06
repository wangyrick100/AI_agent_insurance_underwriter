"""Factory that returns the appropriate LLM based on configuration."""

import config
from .base import BaseLLM
from .mock_llm import MockLLM
from .openai_llm import OpenAILLM


def create_llm() -> BaseLLM:
    """Return a :class:`BaseLLM` instance.

    * If ``config.USE_MOCK_LLM`` is ``True`` (the default when no API key is
      present), a :class:`MockLLM` is returned.
    * Otherwise an :class:`OpenAILLM` is returned, connecting to the model
      specified in ``config.OPENAI_MODEL``.
    """
    if config.USE_MOCK_LLM:
        return MockLLM()
    return OpenAILLM(api_key=config.OPENAI_API_KEY, model=config.OPENAI_MODEL)
