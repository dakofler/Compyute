"""text preprocessing module"""

from __future__ import annotations
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


def remove_punctuation(data: str):
    """Removes punctuation characters from a string.

    Parameters
    ----------
    data : str
        String containing punctuation characters.

    Returns
    ----------
    str
        Clean string.
    """
    punctuations = """!()-[]{};:'",<>./?@#$%^&*_~"""
    data_clean = data
    for punctuation in punctuations:
        data_clean = data_clean.replace(punctuation, "")

    for i in range(5):
        i = 5 - i
        data_clean = data_clean.replace("\n" * i, " ")
    return data_clean


@dataclass
class Tokenizer(ABC):
    """Tokenizer base class."""

    tokens: dict[str, int] = field(default_factory=dict)

    @abstractmethod
    def fit(self, data: str, max_tokens: int = 100) -> None:
        """Fits the tokenizer to data."""


@dataclass(init=False)
class WordTokenizer(Tokenizer):
    """Creates tokens from entire words."""

    def fit(self, data: str, max_tokens: int = 100, dummy: str = "<DUMMY>"):
        """Fits the tokenizer to data.

        Parameters
        ----------
        data : str
            Text the tokens should be extracted from
        max_tokens : int, optional
            Maximum number of tokens to be generated, by default 100
        dummy : str, optional
            Dummy token for unknown words, by default "<DUMMY>"
        """
        data_clean = remove_punctuation(data).lower()
        data_split = data_clean.split(" ")
        # get unique elements
        data_unique = list(set(data_split))
        # sort elements by occurence in data
        data_sorted = sorted(data_unique, key=data_split.count, reverse=True)
        tokens = data_sorted[:max_tokens]
        self.tokens = {token: i + 1 for i, token in enumerate(tokens)}
        self.tokens[dummy] = len(self.tokens) + 1

    def encode(self, string: str) -> int:
        """Encodes a string.

        Parameters
        ----------
        data : str
            String to be encoded.

        Returns
        -------
        int
            Token id.
        """
        return self.tokens.get(string, len(self.tokens))

    def decode(self, token: int) -> str:
        """Decodes an integer.

        Parameters
        ----------
        token : int
            Integer value to be decoded.

        Returns
        -------
        str
            Token.
        """
        return list(filter(lambda x: self.tokens[x] == token, self.tokens))[0]
