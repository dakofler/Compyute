"""Text data preprocessing utilities."""

from __future__ import annotations

import pickle
import re
import string
from abc import ABC, abstractmethod
from typing import Optional

import regex
from tqdm.auto import trange

from ..base_tensor import Tensor, tensor

__all__ = [
    "CharacterTokenizer",
    "WordTokenizer",
    "BPETokenizer",
    "save_tokenizer",
    "load_tokenizer",
]

WORD_PATTERN = r'([,.:;?_!"()\']|--|\s)'
BPE_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class Tokenizer(ABC):
    """Tokenizer base class."""

    def __init__(self, oov_token: str = "") -> None:
        self.oov_token = oov_token
        self.vocab: dict[int, str | bytes] = {}
        self.ivocab: dict[str | bytes, int] = {}

    @property
    def vocab_size(self) -> int:
        """Number of unique tokens."""
        return len(self.vocab)

    def fit(self, text: str, vocab_size: int | None = None) -> None:
        """Fits the tokenizer to text.

        Parameters
        ----------
        text : str
            Text the tokens should be extracted from.
        vocab_size : int | None, optional
            Number of tokens to be generated. Defaults to ``None``.
        """

    @abstractmethod
    def encode(self, text: str) -> Tensor:
        """Encodes text to token ids.

        Parameters
        ----------
        text : str
            Text to be encoded to token ids.

        Returns
        -------
        Tensor
            Tensor of token ids.
        """

    @abstractmethod
    def decode(self, token_ids: Tensor) -> str:
        """Decodes token ids to text..

        Parameters
        ----------
        token_ids : Tensor
            Tensor of integer token ids to be decoded.

        Returns
        -------
        str
            Decoded text.
        """


class CharacterTokenizer(Tokenizer):
    """Creates character tokens."""

    def __init__(self, oov_token: str = "<|unk|>") -> None:
        super().__init__(oov_token)
        all_tokens = [oov_token, "\n"] + list(string.printable)
        self.vocab = dict(enumerate(all_tokens))
        self.ivocab = {token: idx for idx, token in self.vocab.items()}

    def encode(self, text: str) -> Tensor:
        preprocessed = list(text)
        return tensor([self.ivocab[s] if s in self.ivocab else 0 for s in preprocessed])

    def decode(self, token_ids: Tensor) -> str:
        return "".join([self.vocab[i.item()] for i in token_ids])


class WordTokenizer(Tokenizer):
    """Creates word tokens."""

    def __init__(self, oov_token: str = "<|unk|>") -> None:
        super().__init__(oov_token)

    def fit(self, text: str, vocab_size: Optional[int] = None) -> None:
        split = re.split(WORD_PATTERN, text)
        split = [s for s in split if s not in [" ", ""]]
        tokens = [self.oov_token, "\n"] + sorted(list(set(split)))
        if vocab_size is not None:
            tokens = tokens[: vocab_size - 1]
        self.vocab = dict(enumerate(tokens))
        self.ivocab = {t: i for i, t in self.vocab.items()}

    def encode(self, text: str) -> Tensor:
        split = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        split = [s for s in split if s not in [" ", ""]]
        return tensor([self.ivocab[s] if s in self.ivocab else 0 for s in split])

    def decode(self, token_ids: Tensor) -> str:
        text = " ".join([self.vocab[i.item()] for i in token_ids])
        text = re.sub(r'\s+([,.:?!"()\'])', r"\1", text)
        return re.sub(r"\n\s", "\n", text)


class BPETokenizer(Tokenizer):
    """Creates tokens using Byte-Pair-Encoding. Mostly follows the code by Andrjey Karpathy."""

    def __init__(self) -> None:
        super().__init__()
        self._merges = {}
        self._pattern = regex.compile(BPE_PATTERN)

    def fit(self, text: str, vocab_size: int = 256) -> None:
        self.vocab = {idx: bytes([idx]) for idx in range(256)}

        if vocab_size <= 256:
            return

        n_merges = vocab_size - 256

        # split text into chunks according to a regex pattern
        text_chunks = regex.findall(self._pattern, text)

        # encode all chunks
        token_ids = [list(chunk.encode("utf-8")) for chunk in text_chunks]

        for i in trange(n_merges, desc="Merges", unit="merges"):

            # get counts for bigrams
            counts = {}
            for chunk_ids in token_ids:
                self._update_counts(chunk_ids, counts)

            # get most occuring bigram
            if len(counts) == 0:
                print(f"Step {i+1}/{n_merges}. No more possible merges found.")
                break
            bigram = max(counts, key=counts.get)

            # replace occurences of bigram with merge id
            idx = 256 + i
            token_ids = [self._merge(chunk_ids, bigram, idx) for chunk_ids in token_ids]

            self._merges[bigram] = idx
            self.vocab[idx] = self.vocab[bigram[0]] + self.vocab[bigram[1]]

    def _update_counts(self, token_ids, counts=None):
        counts = counts if counts is not None else {}
        for bigram in zip(token_ids, token_ids[1:]):
            counts[bigram] = counts.get(bigram, 0) + 1
        return counts

    def _merge(self, token_ids, bigram, idx):
        new_ids = []
        i = 0

        while i < len(token_ids):
            # if not the last id and the bigram occurs, add new idx
            if (
                i < len(token_ids) - 1
                and token_ids[i] == bigram[0]
                and token_ids[i + 1] == bigram[1]
            ):
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(token_ids[i])
                i += 1

        return new_ids

    def _encode_chunk(self, text_bytes):
        token_ids = list(text_bytes)

        while len(token_ids) >= 2:
            counts = self._update_counts(token_ids)

            # get bigram that first occured in merges
            bigram = min(counts, key=lambda p: self._merges.get(p, float("inf")))
            if bigram not in self._merges:
                break

            idx = self._merges[bigram]
            token_ids = self._merge(token_ids, bigram, idx)
        return token_ids

    def encode(self, text: str) -> Tensor:
        text_chunks = regex.findall(self._pattern, text)
        token_ids = []

        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = self._encode_chunk(chunk_bytes)
            token_ids.extend(chunk_ids)

        return tensor(token_ids)

    def decode(self, token_ids: Tensor) -> str:
        part_bytes = []

        for idx in token_ids:
            idx = idx.item()
            if idx not in self.vocab:
                raise ValueError(f"invalid token id: {idx}")
            part_bytes.append(self.vocab[idx])

        text_bytes = b"".join(part_bytes)
        return text_bytes.decode("utf-8", errors="replace")


def save_tokenizer(tokenizer: Tokenizer, filepath: str) -> None:
    """Saves a tokenizer to a binary file.

    Parameters
    ----------
    filepath : str
        Path where the file should be saved to.
    """
    with open(filepath, "wb") as file:
        pickle.dump(tokenizer, file)


def load_tokenizer(filepath: str) -> Tokenizer:
    """Load a tokenizer from a binary file.

    Parameters
    ----------
    filepath : str
        Path to the binary file to load.

    Returns
    -------
    Tokenizer
        Loaded tokenizer.
    """
    with open(filepath, "rb") as file:
        obj = pickle.load(file)
    return obj
