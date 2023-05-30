![](https://img.shields.io/badge/version-0.0.1-yellow.svg)

**Notes:**
- PACKAGE UNDER CONSTRUCTION!
- Functionality developed as part of the course *Data Collection, Processing & Analysis* and for my master's thesis @ [SODAS, UCPH](https://sodas.ku.dk/).

# politician2vec

Welcome to the `politician2vec` package. Based on massive political text corpora, this package allows for learning, manipulating, and visualising embeddings of words, documents, and politicians in the same high-dimensional semantic space, while simultaneously inferring party positions. Experimental funcitonality allows for measuring latent political concepts by identifying subspaces of the embedding and projecting actors onto these.

The package is adapted from Dimo Angelov's `top2vec` for unsupervised infererence of topics in semantic space (analogous to political parties *in casu*), and its core functionality relies on the `gensim` implementation of Le & Mikolov's `doc2vec`.

Interactive 2D/3D projections are possible, e.g. by exporting vectors and relevant metadata as separate 2D tensors to TensorBoard-compatible .TSV format (notebook-compatible visualisation utilities to be added, please refer to [this](https://github.com/mathiasbruun/DCPA) demo repo in the meantime).

## Package structure

```
politician2vec
│   README.md
│   setup.py    
│   
└───docs                            # Contains documentation and requirements
│   │   LICENSE.txt
│   │   requirements.txt
│   
└───bin
│   │   demo_notebook.ipynb
│   │
│   └───tensorboard                 # Contains example of interactive TensorBoard visualisation
│       │   tb_config.json
│       │   tb_doc_tensor.tsv
│       │   tb_metadata_tensor.tsv
│       │   tb_state.txt
│   
politician2vec functions
│
└───politician2vec                  # Contains politican2vec functionality
│   │   Politician2Vec.py
│   │   utils.py
```

## Requirements

The current version of `politician2vec` was developed in Python 3.8.9. Demos will be provided as notebooks–for now, please refer to [this](https://github.com/mathiasbruun/DCPA) repo for examples of actor positions across various data sourcesn and to [this](https://github.com/mathiasbruun/GeneralisedPoliticalScaling) repo for scaling of political actors by projection to theoretically defined subspaces..

Currently known package-specific Python dependencies:

- `numpy` >= 1.20.0
- `pandas`
- `gensim` >= 4.0.0
- `umap-learn` >= 0.5.1
- `hdbscan` >= 0.8.27
- `wordcloud`
- `tensorflow`
- `tensorflow_hub`
- `tensorflow_text`
- `torch`
- `sentence_transformers`
- `hnswlib`

## Installation

Install politician2vec via command line:

`$ pip install git+https://git@github.com/mathiasbruun/politician2vec.git`

## Example import

```python
from politician2vec import Politician2Vec
from politician2vec.utils import * # please refer to the utils module for further elaboration
```

## Demos

Notebooks demonstrating use cases and workflows to be added in `bin/notebooks/`.

TensorBoard demo visualisations also coming soon.

## Foundational literature

- Angelov, D. (2020). *Top2Vec: Distributed Representations of Topics*. arXiv: [2008.09470](https://arxiv.org/abs/2008.09470).
- Le, Q. V., & T. Mikolov (2014). *Distributed Representations of Sentences and Documents*. arXiv: [1405.4053](https://doi.org/10.48550/arXiv.1405.4053).
- Mikolov, T., K. Chen, G. Corrado & J. Dean (2013). *Efficient Estimation of Word Representations in Vector Space*. arXiv: [1301.3781](https://arxiv.org/abs/1301.3781).