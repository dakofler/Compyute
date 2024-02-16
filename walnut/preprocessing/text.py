"""text preprocessing module"""

from __future__ import annotations
from abc import ABC, abstractmethod
import re

from walnut.tensor import Tensor

__all__ = ["CharacterTokenizer", "WordTokenizer"]


class Tokenizer(ABC):
    """Tokenizer base class."""

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.oov_token: str = ""

    @property
    def vocab_size(self) -> int:
        """Number of tokens."""
        return len(self.vocab)

    @abstractmethod
    def fit(self, text: str, max_tokens: int | None = None) -> None:
        """Fits the tokenizer to text."""

    @abstractmethod
    def encode(self, text: str) -> Tensor:
        """Encodes text."""

    @abstractmethod
    def decode(self, token_ids: Tensor) -> str:
        """Decodes token ids."""


class CharacterTokenizer(Tokenizer):
    """Creates character tokens."""

    def __init__(self, oov_token: str = "<|unk|>") -> None:
        super().__init__()
        self.oov_token = oov_token
        all_tokens = [oov_token, "\n"] + list(
            "!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~ "
        )
        self.vocab = {t: i for i, t in enumerate(all_tokens)}

    def fit(self, text: str, max_tokens: int | None = None) -> None:
        """Fits the tokenizer to text.

        Parameters
        ----------
        text : str
            Text the tokens should be extracted from.
        max_tokens : int, optional
            Maximum number of tokens to be generated, by default 100.
        """

    def encode(self, text: str) -> Tensor:
        """Encodes text.

        Parameters
        ----------
        text : str
            Text to be encoded.

        Returns
        -------
        Tensor
            Tensor of token ids.
        """
        preprocessed = list(text)
        ids = [self.vocab[s] if s in self.vocab else 0 for s in preprocessed]
        return Tensor(ids, dtype="int")

    def decode(self, token_ids: Tensor) -> str:
        """Decodes token ids.

        Parameters
        ----------
        tokens : Tensor
            Tensor of token_ids to be decoded.

        Returns
        -------
        str
            Text.
        """
        id_to_tokens = {i: t for t, i in self.vocab.items()}
        return "".join([id_to_tokens[int(i.item())] for i in token_ids])


class WordTokenizer(Tokenizer):
    """Creates character tokens."""

    def __init__(self, oov_token: str = "<|unk|>") -> None:
        super().__init__()
        self.oov_token = oov_token
        self.vocab = {}

    def fit(self, text: str, max_tokens: int | None = None) -> None:
        """Fits the tokenizer to text.

        Parameters
        ----------
        text : str
            Text the tokens should be extracted from.
        max_tokens : int | None, optional
            Maximum number of tokens to be generated, by default None.
        """
        split = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        split = [s for s in split if s not in [" ", ""]]
        tokens = [self.oov_token, "\n"] + sorted(list(set(split)))
        if max_tokens is not None:
            tokens = tokens[: max_tokens - 1]
        self.vocab = {t: i for i, t in enumerate(tokens)}

    def encode(self, text: str) -> Tensor:
        """Encodes a text.

        Parameters
        ----------
        text : str
            Text to be encoded.

        Returns
        -------
        Tensor
            Tensor of token ids.
        """
        split = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        split = [s for s in split if s not in [" ", ""]]
        ids = [self.vocab[s] if s in self.vocab else 0 for s in split]
        return Tensor(ids, dtype="int")

    def decode(self, token_ids: Tensor) -> str:
        """Decodes token ids.

        Parameters
        ----------
        tokens : Tensor
            Tensor of token ids to be decoded.

        Returns
        -------
        str
            Concatenated tokens.
        """
        id_to_tokens = {i: t for t, i in self.vocab.items()}
        text = " ".join([id_to_tokens[int(i.item())] for i in token_ids])
        text = re.sub(r'\s+([,.:?!"()\'])', r"\1", text)
        return re.sub(r"\n\s", "\n", text)
