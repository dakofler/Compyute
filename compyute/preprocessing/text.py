"""text preprocessing module"""

from __future__ import annotations
from abc import ABC, abstractmethod
import pickle
import re

import regex
from tqdm.auto import trange
from compyute.tensor import Tensor

__all__ = [
    "CharacterTokenizer",
    "WordTokenizer",
    "BPETokenizer",
    "save_tokenizer",
    "load_tokenizer",
]

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class Tokenizer(ABC):
    """Tokenizer base class."""

    def __init__(self) -> None:
        self.oov_token: str = ""
        self._vocab: dict[str, int] = {}
        self._ivocab: dict[int, str] = {}

    @property
    def vocab_size(self) -> int:
        """Number of tokens."""
        return len(self._vocab)

    def fit(self, text: str, vocab_size: int | None = None) -> None:
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
        self._vocab = {t: i for i, t in enumerate(all_tokens)}
        self._ivocab = {i: t for t, i in self._vocab.items()}

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
        ids = [self._vocab[s] if s in self._vocab else 0 for s in preprocessed]
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
        return "".join([self._ivocab[i] for i in token_ids])


class WordTokenizer(Tokenizer):
    """Creates word tokens."""

    def __init__(self, oov_token: str = "<|unk|>") -> None:
        super().__init__()
        self.oov_token = oov_token

    def fit(self, text: str, vocab_size: int | None = None) -> None:
        """Fits the tokenizer to text.

        Parameters
        ----------
        text : str
            Text the tokens should be extracted from.
        vocab_size : int | None, optional
            Number of tokens to be generated, by default None.
        """
        split = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        split = [s for s in split if s not in [" ", ""]]
        tokens = [self.oov_token, "\n"] + sorted(list(set(split)))
        if vocab_size is not None:
            tokens = tokens[: vocab_size - 1]
        self._vocab = {t: i for i, t in enumerate(tokens)}
        self._ivocab = {i: t for t, i in self._vocab.items()}

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
        ids = [self._vocab[s] if s in self._vocab else 0 for s in split]
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
        text = " ".join([self._ivocab[i] for i in token_ids])
        text = re.sub(r'\s+([,.:?!"()\'])', r"\1", text)
        return re.sub(r"\n\s", "\n", text)


class BPETokenizer(Tokenizer):
    """Creates tokens using Byte-Pair-Encoding. Mostly follows the code by Andrjey Karpathy."""

    def __init__(self) -> None:
        super().__init__()
        self._vocab = {}
        self._merges = {}
        self._pattern = regex.compile(GPT4_SPLIT_PATTERN)

    def fit(self, text: str, vocab_size: int = 256) -> None:
        """Fits the tokenizer to text.

        Parameters
        ----------
        text : str
            Text the tokens should be extracted from.
        vocab_size : int | None, optional
            Number of tokens to be generated, by default 256.
        """
        self._vocab = {idx: bytes([idx]) for idx in range(256)}

        if vocab_size <= 256:
            return

        num_merges = vocab_size - 256

        # split text into chunks according to a regex pattern
        text_chunks = regex.findall(self._pattern, text)

        # encode all chunks
        token_ids = [list(chunk.encode("utf-8")) for chunk in text_chunks]

        for i in trange(num_merges, desc="Merges"):

            # get counts for bigrams
            counts = {}
            for chunk_ids in token_ids:
                self._update_counts(chunk_ids, counts)

            # get most occuring bigram
            if len(counts) == 0:
                print(f"Step {i+1}/{num_merges}. No more possible merges found.")
                break
            bigram = max(counts, key=counts.get)

            # replace occurences of bigram with merge id
            idx = 256 + i
            token_ids = [self._merge(chunk_ids, bigram, idx) for chunk_ids in token_ids]

            self._merges[bigram] = idx
            self._vocab[idx] = self._vocab[bigram[0]] + self._vocab[bigram[1]]

    def _update_counts(self, token_ids, counts=None):
        counts = {} if counts is None else counts
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
        text_chunks = regex.findall(self._pattern, text)
        token_ids = []

        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = self._encode_chunk(chunk_bytes)
            token_ids.extend(chunk_ids)

        return Tensor(token_ids, dtype="int32")

    def decode(self, token_ids: Tensor) -> str:
        """Decodes token ids.

        Parameters
        ----------
        tokens : Tensor
            Tensor of token ids to be decoded.

        Returns
        -------
        str
            Decoded string.
        """
        part_bytes = []

        for idx in token_ids:
            if idx in self._vocab:
                part_bytes.append(self._vocab[idx])
            else:
                raise ValueError(f"invalid token id: {idx}")

        text_bytes = b"".join(part_bytes)
        return text_bytes.decode("utf-8", errors="replace")


def save_tokenizer(tokenizer: Tokenizer, filepath: str) -> None:
    """Saves a tokenizer to a binary file.

    Parameters
    ----------
    filepath : str
        Path to the file.
    """
    file = open(filepath, "wb")
    pickle.dump(tokenizer, file)
    file.close()


def load_tokenizer(filepath: str) -> Tokenizer:
    """Load a tokenizer from a previously saved binary file.

    Parameters
    ----------
    filepath : str
        Path to the file.

    Returns
    -------
    Tokenizer
        Loaded tokenizer.
    """
    file = open(filepath, "rb")
    obj = pickle.load(file)
    file.close()
    return obj
