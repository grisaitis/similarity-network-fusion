# Similarity network fusion (SNF)

This repo is an implementation of the paper

>[Wang, Bo, et al. "Similarity network fusion for aggregating data types on a genomic scale." Nature methods 11.3 (2014): 333.](http://sci-hub.se/10.1038/nmeth.2810)

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

### My thoughts (questions) on the paper

Why did they design this method the way it is? Why not just take the mean or median edge weight across networks? 

The methodology employs some sparsity in how it iteratively reweights similarity values. Related to the first question - why is this done? What does it achieve.

How is the neighborhood size K tuned?

The algorithm assumes, for a given data type (e.g. DNA methyltion), that all components of the feature vector for that type are equally important. (The similarity metric uses the euclidean distance in feature space of that data type.) Is that desirable? Can that be improved?

Related to this last question: can the distance function, or similarity kernel, be learned? 

Also, why does μ exist in the kernel function? They recommend setting it in `[0.3, 0.8]`. How much does it impact results? 

What if a different kernel function were used? The usage of μ suggests that kernel values are too low. Usage of a kernel like Matern 5/2 would increase kernel values for things that are farther away, which might or might not be "better."

What are the evaluation criteria of results?
