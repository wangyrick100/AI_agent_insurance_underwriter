"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod
from typing import Optional


class BaseLLM(ABC):
    """Minimal interface that every LLM provider must implement."""

    @abstractmethod
    def complete(self, prompt: str, system: Optional[str] = None) -> str:
        """Send *prompt* to the model and return the completion as a string.

        Parameters
        ----------
        prompt:
            The user-facing prompt / question.
        system:
            Optional system-level instruction that frames how the model
            should behave (analogous to OpenAI's ``system`` role).

        Returns
        -------
        str
            The model's text response.
        """
