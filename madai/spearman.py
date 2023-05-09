from collections import Counter
from functools import partial
from random import sample
from typing import Callable, Union

import sienna
import typer
from scipy.stats import spearmanr

from madai.utils import avg, get_tokenizer, std, tokenize


class Corpus:
    def __init__(
        self,
        tokenizer: Callable[[str], list[str]],
        docs: list[str],
        slice_size: Union[int, float],
    ):
        self.doc_n: int = len(docs)
        self.slice_size: int = max(
            slice_size if isinstance(slice_size, int) else int(self.doc_n * slice_size),
            1,
        )
        self.tokenized_docs: list[list[str]] = [tokenizer(doc) for doc in docs]
        self.slices: list[list[list[str]]] = [
            self.tokenized_docs[i : i + self.slice_size]
            for i in range(0, self.doc_n, self.slice_size)
        ]

    def sample_docs(self) -> list[list[str]]:
        sampled_slices = sample(self.slices, self.doc_n // 2)
        return [doc for slice in sampled_slices for doc in slice]

    def sample_words(self) -> list[str]:
        docs = self.sample_docs()
        return [word for doc in docs for word in doc]


def sim(a_words: list[str], b_words: list[str], top_n_words: int) -> float:
    a_ranked_words = [w for w, _ in Counter(a_words).most_common(top_n_words)]
    b_ranked_words = [w for w, _ in Counter(b_words).most_common(top_n_words)]

    assert isinstance(a_ranked_words[0], str)
    assert isinstance(b_ranked_words[0], str)

    d = spearmanr(a_ranked_words, b_ranked_words)[0]

    return d


def sim_corpus(a: Corpus, b: Corpus, top_n_words: int) -> float:
    return sim(a.sample_words(), b.sample_words(), top_n_words)


app = typer.Typer()


@app.command()
def spearman(
    a: str = typer.Argument(..., help="Path to a corpus"),
    b: str = typer.Argument(..., help="Path to another corpus"),
    lang: str = typer.Option("en", help="Language code for both corpus."),
    top_n_words: int = typer.Option(
        500, help="Number of most frequent words to consider."
    ),
    iter_n: int = typer.Option(50, help="Number of iteration to take avg from."),
    slice_size: float = typer.Option(0.25, help="Size of each slice."),
):
    """
    Use the Spearman rank-order correlation coefficient to compare two corpus.
    """
    tok = get_tokenizer(lang)
    tokenizer = partial(tokenize, tok, False)

    adocs = sienna.load(a)
    bdocs = sienna.load(b)
    assert isinstance(adocs, list)
    assert isinstance(bdocs, list)

    ac = Corpus(tokenizer, adocs, slice_size)
    bc = Corpus(tokenizer, bdocs, slice_size)

    scores = []

    for _ in range(iter_n):
        score = sim_corpus(ac, bc, top_n_words)
        scores.append(score)

    print(f"{avg(scores)}±{std(scores)}")
