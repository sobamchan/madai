# madai

Compute difference between two corpus by using chi2.
Implementation is based on [Measures for Corpus Similarity and Homogeneity](https://aclanthology.org/W98-1506).

I am not fully sure if this implementation is perfectly follow this paper.
Feel free to make issues to point out some problems if you find.

## Installation

```
pip install madai
```

## Usage

madai implements two ways of computing similarity between two corpus, chi2 and spearman.
Use spearman when two corpus are different in size.

Two target corpus need to be text files, each line containing one document/sentence.

```
madai chi2 /path/to/corpus/a /path/to/corpus/b

# or

madai spearman /path/to/corpus/a /path/to/corpus/b
```

To view parameters, run,
```
madai --help
```
