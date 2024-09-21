"""Text data preprocessing utilities."""

from __future__ import annotations

import re
import string
from abc import ABC, abstractmethod
from collections import Counter, OrderedDict
from itertools import pairwise
from typing import Optional

import regex
from tqdm.auto import trange

__all__ = ["CharacterTokenizer", "WordTokenizer", "BPETokenizer"]

WORD_PATTERN = r'([,.:;?_!"()\']|--|\s)'
BPE_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class Tokenizer(ABC):
    """Tokenizer base class."""

    oov_token: str
    vocab: dict
    ivocab: dict

    def __init__(
        self,
        oov_token: str = "",
        vocab: Optional[dict] = None,
        ivocab: Optional[dict] = None,
    ) -> None:
        self.oov_token = oov_token
        self.vocab = vocab or {}
        self.ivocab = ivocab or {}

    @property
    def vocab_size(self) -> int:
        """Number of unique tokens."""
        return len(self.vocab)

    def fit(self, text: str, vocab_size: Optional[int] = None) -> None:
        """Fits the tokenizer to text.

        Parameters
        ----------
        text : str
            Text the tokens should be extracted from.
        vocab_size : int | None, optional
            Number of tokens to be generated. Defaults to ``None``.
        """

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Encodes text to token ids.

        Parameters
        ----------
        text : str
            Text to be encoded to token ids.

        Returns
        -------
        list[int]
            List of token ids.
        """

    @abstractmethod
    def decode(self, token_ids: list[int]) -> str:
        """Decodes token ids to text..

        Parameters
        ----------
        token_ids : list[int]
            List of integer token ids to be decoded.

        Returns
        -------
        str
            Decoded text.
        """

    def get_state_dict(self) -> OrderedDict:
        """Returns the tokenizer state dictionary."""
        return OrderedDict(
            oov_token=self.oov_token, vocab=self.vocab, ivocab=self.ivocab
        )

    def load_state_dict(self, state_dict: OrderedDict) -> None:
        """Loads the tokenizer state from a state dict.

        Parameters
        ----------
        state_dict : OrderedDict
            State dict containing parameters and buffers.
        """
        for k, v in state_dict.items():
            setattr(self, k, v)


class CharacterTokenizer(Tokenizer):
    """Uses single characters as tokens. Characters are taken from ``string.printable``."""

    def __init__(self, oov_token: str = "<|unk|>") -> None:
        all_tokens = [oov_token, "\n"] + list(string.printable)
        vocab = dict(enumerate(all_tokens))
        ivocab = {token: idx for idx, token in vocab.items()}
        super().__init__(oov_token, vocab, ivocab)

    def encode(self, text: str) -> list[int]:
        preprocessed = list(text)
        return [self.ivocab[s] if s in self.ivocab else 0 for s in preprocessed]

    def decode(self, token_ids: list[int]) -> str:
        return "".join([self.vocab[i] for i in token_ids])


class WordTokenizer(Tokenizer):
    """Uses whole words as tokens."""

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

    def encode(self, text: str) -> list[int]:
        split = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        split = [s for s in split if s not in [" ", ""]]
        return [self.ivocab[s] if s in self.ivocab else 0 for s in split]

    def decode(self, token_ids: list[int]) -> str:
        text = " ".join([self.vocab[i] for i in token_ids])
        text = re.sub(r'\s+([,.:?!"()\'])', r"\1", text)
        return re.sub(r"\n\s", "\n", text)


class BPETokenizer(Tokenizer):
    """Uses learned sub-words as tokens by applying the Byte-Pair Encoding algorithm as described by
    `Sennrich et al., 2016 <https://arxiv.org/pdf/1508.07909>`_.
    Mostly follows Andrjey Karpathy's `minbpe <https://github.com/karpathy/minbpe>`_.
    """

    _merges: dict
    _pattern: regex.Pattern

    def __init__(self, oov_token: str = "<|unk|>") -> None:
        super().__init__(oov_token)
        self._merges = {}
        self._pattern = regex.compile(BPE_PATTERN)

    def fit(self, text: str, vocab_size: Optional[int] = None) -> None:
        vocab_size = vocab_size or 256
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
            counts: Counter[tuple[int, int]] = Counter()
            for chunk in token_ids:
                chunk_counts = Counter(pairwise(chunk))
                for bigram, count in chunk_counts.items():
                    counts[bigram] = counts.get(bigram, 0) + count

            if len(counts) == 0:
                print(f"Step {i+1}/{n_merges}. No more possible merges found.")
                break

            # get most occuring bigram
            bigram = counts.most_common(1)[0][0]

            # replace occurences of bigram with merge id
            idx = 256 + i
            token_ids = [self._merge(chunk_ids, bigram, idx) for chunk_ids in token_ids]

            self._merges[bigram] = idx
            self.vocab[idx] = self.vocab[bigram[0]] + self.vocab[bigram[1]]

    def _merge(
        self, token_ids: list[int], bigram: tuple[int, int], idx: int
    ) -> list[int]:
        new_ids: list[int] = []
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

    def _encode_chunk(self, text_bytes: bytes) -> list[int]:
        token_ids = list(text_bytes)

        while len(token_ids) >= 2:
            counts = Counter(pairwise(token_ids))

            # get bigram that first occured in merges
            bigram = min(counts, key=lambda p: self._merges.get(p, float("inf")))
            if bigram not in self._merges:
                break

            idx = self._merges[bigram]
            token_ids = self._merge(token_ids, bigram, idx)
        return token_ids

    def encode(self, text: str) -> list[int]:
        text_chunks: list[str] = regex.findall(self._pattern, text)
        token_ids = []

        for chunk in text_chunks:
            chunk_ids = self._encode_chunk(chunk.encode("utf-8"))
            token_ids.extend(chunk_ids)

        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        part_bytes = []

        for idx in token_ids:
            if idx not in self.vocab:
                raise ValueError(f"invalid token id: {idx}")
            part_bytes.append(self.vocab[idx])

        text_bytes = b"".join(part_bytes)
        return text_bytes.decode("utf-8", errors="replace")

    def get_state_dict(self) -> OrderedDict:
        """Returns the tokenizer state dictionary."""
        return OrderedDict(
            oov_token=self.oov_token,
            vocab=self.vocab,
            ivocab=self.ivocab,
            merges=self._merges,
            pattern=self._pattern,
        )
