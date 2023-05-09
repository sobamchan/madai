from collections import Counter
from dataclasses import dataclass, field
from functools import partial
from random import sample
from typing import Union

import numpy as np
import scipy.stats as stats
import sienna
import typer
from spacy.tokenizer import Tokenizer
from tqdm import tqdm

from madai.utils import get_tokenizer, tokenize


@dataclass
class FreqCounter:
    seed_words: set[str]
    w2idx: dict[str, int] = field(init=False)
    counter: list[int] = field(init=False)

    def __post_init__(self):
        self.w2idx = {w: i for i, w in enumerate(self.seed_words)}
        self.counter = [1 for _ in range(len(self.seed_words))]

    def __len__(self):
        return len(self.counter)

    def init_counter(self) -> None:
        self.counter = [1 for _ in range(len(self.seed_words))]

    def load_words(self, words: list[str]) -> None:
        for w in words:
            if w in self.w2idx:
                self.counter[self.w2idx[w]] += 1


def sim(a_cnts: FreqCounter, b_cnts: FreqCounter) -> float:
    assert len(a_cnts) == len(b_cnts)
    obs1: list[int]
    exp1: list[float]
    obs2: list[int]
    exp2: list[float]

    obs1 = a_cnts.counter
    obs2 = b_cnts.counter

    n1 = sum(a_cnts.counter)
    n2 = sum(b_cnts.counter)

    exp1 = [(n1 * (obs1[i] + obs1[i])) / (n1 + n2) for i in range(len(a_cnts.counter))]
    exp2 = [(n2 * (obs1[i] + obs1[i])) / (n1 + n2) for i in range(len(a_cnts.counter))]

    return stats.chi2_contingency([[obs1, exp1], [obs2, exp2]]).statistic


def _run(
    a: list[str],
    b: list[str],
    tok: Tokenizer,
    slice_size: Union[int, float],
    top_n: int = 500,
    n_iter: int = 5,
    remove_stopwords: bool = False,
) -> np.ndarray:
    a_doc_n = len(a)
    b_doc_n = len(b)

    doc_n = min(a_doc_n, b_doc_n)
    a = a[:doc_n]
    b = b[:doc_n]

    print(f"Use {doc_n} documents for each corpus.")

    # Number of sample for each slice
    slice_size = slice_size if isinstance(slice_size, int) else int(doc_n * slice_size)
    slice_size = max(1, slice_size)

    print(f"Slice size: {slice_size}")

    # Tokenization for freq word counting
    tokenizer = partial(tokenize, tok, remove_stopwords)
    a_words = [word for doc in a for word in tokenizer(doc)]
    b_words = [word for doc in b for word in tokenizer(doc)]

    assert isinstance(a_words, list)
    assert isinstance(a_words[0], str)

    # Find Top N words
    freq_words = set(
        [word for word, _ in Counter(a_words + b_words).most_common(top_n)]
    )
    assert len(freq_words) <= top_n

    a_fc = FreqCounter(seed_words=freq_words)
    b_fc = FreqCounter(seed_words=freq_words)

    # Split corpus into slices
    a_sliices = [a[i : i + slice_size] for i in range(0, doc_n, slice_size)]
    b_sliices = [b[i : i + slice_size] for i in range(0, doc_n, slice_size)]

    # A list of scores, further taken avg, std
    scores: list[float] = []

    # A number of slices to be used for each iter (50% of a corpus)
    n_slices_to_sample = min(len(a_sliices), len(b_sliices)) // 2

    for _ in tqdm(range(n_iter), total=n_iter):
        a_fc.init_counter()
        b_fc.init_counter()

        sub_a = [
            doc for slice in sample(a_sliices, k=n_slices_to_sample) for doc in slice
        ]
        sub_b = [
            doc for slice in sample(b_sliices, k=n_slices_to_sample) for doc in slice
        ]

        assert isinstance(sub_a[0], str)
        assert isinstance(sub_b[0], str)

        a_words = [word for doc in a for word in tokenizer(doc)]
        b_words = [word for doc in b for word in tokenizer(doc)]

        a_fc.load_words(a_words)
        b_fc.load_words(b_words)

        score = sim(a_fc, b_fc)
        scores.append(score)

    return np.array(scores)


app = typer.Typer()


@app.command()
def chi2(
    a: str = typer.Argument(..., help="Path to a corpus"),
    b: str = typer.Argument(..., help="Path to another corpus"),
    lang: str = typer.Option("en", help="Language code for both corpus."),
    slice_size: float = typer.Option(0.25, help="Size of each slice."),
    iter_n: int = typer.Option(50, help="Number of iteration to take avg from."),
    top_n_words: int = typer.Option(
        500, help="Number of most frequent words to consider."
    ),
    remove_stopwords: bool = False,
):
    if remove_stopwords:
        assert lang == "en", "Stopword remove is only supported for English (en)"

    a_texts = sienna.load(a)
    b_texts = sienna.load(b)

    assert isinstance(a_texts, list)
    assert isinstance(b_texts, list)

    tok = get_tokenizer(lang)

    scores = _run(
        a_texts, b_texts, tok, slice_size, top_n_words, iter_n, remove_stopwords
    )
    score_avg = np.average(scores)
    score_std = np.std(scores)
    print(f"Distance between corpus: {score_avg:.2f}±{score_std:.2f}")
