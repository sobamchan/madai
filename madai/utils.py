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
