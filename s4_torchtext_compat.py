"""Small pure-Python subset of torchtext used by the official S4 LRA loader.

Torchtext 0.18 is the final torchtext release and cannot load against newer
PyTorch ABIs.  S4's resolved LRA configurations only need vocabulary building;
the compatibility modules below preserve that behavior without importing the
retired C++ extension.
"""

from __future__ import annotations

import csv
import importlib.machinery
import sys
from collections import Counter, OrderedDict
from functools import partial
from types import ModuleType
from typing import Iterable, List, Optional, Sequence


class CompatVocab:
    """Python implementation of the torchtext Vocab API used by S4."""

    def __init__(self, tokens: Sequence[str]) -> None:
        self._itos = list(tokens)
        self._stoi = {token: index for index, token in enumerate(self._itos)}
        self._default_index: Optional[int] = None

    def __len__(self) -> int:
        return len(self._itos)

    def __contains__(self, token: str) -> bool:
        return token in self._stoi

    def __getitem__(self, token: str) -> int:
        index = self._stoi.get(token, self._default_index)
        if index is None:
            raise RuntimeError(f"Token {token!r} not found and default index is not set")
        return index

    def __call__(self, tokens: List[str]) -> List[int]:
        return self.lookup_indices(tokens)

    def set_default_index(self, index: Optional[int]) -> None:
        self._default_index = index

    def get_default_index(self) -> Optional[int]:
        return self._default_index

    def lookup_indices(self, tokens: List[str]) -> List[int]:
        return [self[token] for token in tokens]

    def lookup_token(self, index: int) -> str:
        if index < 0 or index >= len(self._itos):
            raise RuntimeError(f"Vocab index {index} is out of range")
        return self._itos[index]

    def lookup_tokens(self, indices: List[int]) -> List[str]:
        return [self.lookup_token(index) for index in indices]

    def get_stoi(self):
        return dict(self._stoi)

    def get_itos(self):
        return list(self._itos)

    def insert_token(self, token: str, index: int) -> None:
        if token in self._stoi:
            raise RuntimeError(f"Token {token!r} already exists")
        if index < 0 or index > len(self._itos):
            raise RuntimeError(f"Vocab index {index} is out of range")
        self._itos.insert(index, token)
        self._stoi = {value: i for i, value in enumerate(self._itos)}

    def append_token(self, token: str) -> None:
        self.insert_token(token, len(self._itos))


def build_vocab_from_iterator(
    iterator: Iterable[Iterable[str]],
    min_freq: int = 1,
    specials: Optional[List[str]] = None,
    special_first: bool = True,
    max_tokens: Optional[int] = None,
) -> CompatVocab:
    """Match torchtext 0.18's frequency and lexical token ordering."""

    counter: Counter[str] = Counter()
    for tokens in iterator:
        counter.update(tokens)

    specials = list(specials or [])
    sorted_tokens = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    if max_tokens is not None:
        if len(specials) >= max_tokens:
            raise AssertionError(
                "len(specials) must be smaller than max_tokens"
            )
        sorted_tokens = sorted_tokens[: max_tokens - len(specials)]

    ordered = OrderedDict(sorted_tokens)
    for token in specials:
        ordered.pop(token, None)
    tokens = [token for token, frequency in ordered.items() if frequency >= min_freq]
    tokens = specials + tokens if special_first else tokens + specials
    return CompatVocab(tokens)


def _split_tokenizer(text: str) -> List[str]:
    return text.split()


def _spacy_tokenize(text: str, spacy_model) -> List[str]:
    return [token.text for token in spacy_model.tokenizer(text)]


def get_tokenizer(tokenizer, language: str = "en"):
    if tokenizer is None:
        return _split_tokenizer
    if callable(tokenizer):
        return tokenizer
    if tokenizer == "spacy":
        import spacy

        return partial(_spacy_tokenize, spacy_model=spacy.load(language))
    raise ValueError(
        f"Tokenizer {tokenizer!r} is outside the S4 LRA compatibility subset"
    )


def unicode_csv_reader(unicode_csv_data, **kwargs):
    return csv.reader(unicode_csv_data, **kwargs)


def _new_module(name: str, *, package: bool = False) -> ModuleType:
    module = ModuleType(name)
    module.__package__ = name if package else name.rpartition(".")[0]
    module.__spec__ = importlib.machinery.ModuleSpec(
        name, loader=None, is_package=package
    )
    if package:
        module.__path__ = []
    return module


def install_torchtext_compat() -> ModuleType:
    """Install an in-process torchtext module containing the S4 LRA subset."""

    for name in tuple(sys.modules):
        if name == "torchtext" or name.startswith("torchtext."):
            del sys.modules[name]

    torchtext = _new_module("torchtext", package=True)
    data = _new_module("torchtext.data", package=True)
    data_utils = _new_module("torchtext.data.utils")
    vocab = _new_module("torchtext.vocab", package=True)
    text_utils = _new_module("torchtext.utils")

    data_utils.get_tokenizer = get_tokenizer
    data.get_tokenizer = get_tokenizer
    data.utils = data_utils
    vocab.Vocab = CompatVocab
    vocab.build_vocab_from_iterator = build_vocab_from_iterator
    text_utils.unicode_csv_reader = unicode_csv_reader

    torchtext.__version__ = "s4-lra-compat"
    torchtext.data = data
    torchtext.vocab = vocab
    torchtext.utils = text_utils

    sys.modules.update(
        {
            "torchtext": torchtext,
            "torchtext.data": data,
            "torchtext.data.utils": data_utils,
            "torchtext.vocab": vocab,
            "torchtext.utils": text_utils,
        }
    )
    return torchtext


def ensure_torchtext_for_s4() -> Optional[Exception]:
    """Use installed torchtext when compatible, otherwise install the subset."""

    try:
        __import__("torchtext")
    except Exception as exc:
        install_torchtext_compat()
        return exc
    return None