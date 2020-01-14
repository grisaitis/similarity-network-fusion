# Similarity Network Fusion (SNF)

This repo implements the paper

>Wang, Bo, et al. "Similarity network fusion for aggregating data types on a genomic scale." Nature methods 11.3 (2014): 333.

available at [nature.com](https://www.nature.com/articles/nmeth.2810) or [sci-hub](http://sci-hub.se/10.1038/nmeth.2810).

## Installation

Requires
* Python 3.8 (or [pyenv](https://github.com/pyenv/pyenv) which will install that automatically)
* [poetry](https://github.com/python-poetry/poetry) for installing dependencies

```
# install dependencies, make virtualenv
poetry install
# do stuff
poetry run jupyter notebook
```

## The paper

The similarity network fusion (SNF) algorithm merges multiple similarity networks (aka affinity graphs) into one. Input graphs share the same vertices, are undirected, are fully connected, and have scalar-valued similarity values. Networks differ in their edge weights (similarity values). The goal is to consolidate these into one network, with new scalar edge weights.

In the context of the paper, vertices are people with cancer, and networks correspond to different types of omic measurements: DNA methylation, RNA, and microRNA expression. Each of these measurement classes or "data types" as referred to in the paper yields different similarity values between people. The algorithm takes data for each of the data types, constructs similarity networks for each, and then "fuses" these networks into one network. The edge weights of this output network combine similarity information across data types.
