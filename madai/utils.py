import numpy as np
from nltk.corpus import stopwords
from spacy.lang.ar import Arabic
from spacy.lang.bg import Bulgarian
from spacy.lang.de import German
from spacy.lang.el import Greek
from spacy.lang.en import English
from spacy.lang.es import Spanish
from spacy.lang.fr import French
from spacy.lang.hi import Hindi
from spacy.lang.ru import Russian
# from spacy.lang.th import Thai
from spacy.lang.tr import Turkish
from spacy.lang.ur import Urdu
from spacy.lang.vi import Vietnamese
from spacy.lang.zh import Chinese
from spacy.tokenizer import Tokenizer

stopwords = stopwords.words("english")


def get_tokenizer(lang: str) -> Tokenizer:
    # No Swahili, Thai
    nlp = {
        "en": English(),
        "ar": Arabic(),
        "bg": Bulgarian(),
        "de": German(),
        "el": Greek(),
        "es": Spanish(),
        "fr": French(),
        "hi": Hindi(),
        "ru": Russian(),
        "tr": Turkish(),
        "ur": Urdu(),
        "vi": Vietnamese(),
        "zh": Chinese(),
    }[lang]
    return nlp.tokenizer


def tokenize(tok: Tokenizer, remove_stopwords: bool, text: str) -> list[str]:
    if remove_stopwords:
        return [
            word.text.lower()
            for word in tok(text)
            if word.text.lower() not in stopwords
        ]
    else:
        return [word.text.lower() for word in tok(text)]


def avg(values: list) -> float:
    return np.mean(np.array(values))


def std(values: list) -> float:
    return np.std(np.array(values))
