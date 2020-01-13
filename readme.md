# Similarity network fusion (SNF)

This repo is an implementation of

>[Wang, Bo, et al. "Similarity network fusion for aggregating data types on a genomic scale." Nature methods 11.3 (2014): 333.](http://sci-hub.se/10.1038/nmeth.2810)

## Installation

Requires
* **Python 3.8**. 
  * This is used automagically if you have [pyenv](https://github.com/pyenv/pyenv).
* [poetry](https://github.com/python-poetry/poetry) for installing dependencies

If running the notebooks here
* install [pyenv]() and poetry if you don't have them
* `pyenv shell 3.8.0`
* `poetry install`

and then `poetry run jupyter notebook` or something for running notebooks.

I recommend installing pyenv and poetry and running `pyenv 

`pip install /path/to/this/project` will work. 

I' [`poetry`](https://github.com/python-poetry/poetry) to m

This algorithm merges multiple similarity networks (aka affinity graphs) into one network. The input graphs share the same vertices, are undirected, are fully connected, and have scalar-valued similarity values. Networks differ in their edge weight (similarity values). The goal is to combine these into one network with one similarity value between verticies.

In the context of the paper, vertices are people with cancer, and networks correspond to different types of omic measurements: DNA methylation, RNA, and microRNA expression. Each of these measurement classes or "data types" as referred to in the paper yields different similarity values between people. The algorithm makes one network with one similarity value that combines information from all input networks.

### My thoughts (questions, really) on the paper

Why did they develop this method for combining networks? Why not just take the mean or median value? 

The methodology employs some sparsity in how it iteratively reweights similarity values. Related to the first question - why is this done?

How is the neighborhood size K tuned?
