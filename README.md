![](https://img.shields.io/badge/version-0.0.1-yellow.svg)

**Notes:**
- PACKAGE UNDER CONSTRUCTION!

# politician2vec

Welcome to the `politician2vec` package. Based on massive political text corpora, this package allows for learning, manipulating, and visualising embeddings of words, documents, and politicians in the same high-dimensional semantic space, while simultaneously inferring party positions.

Core functionality relies heavily on the `gensim` implementation of Le & Mikolov's `doc2vec`, and the package is adapted from Dimo Angelov's `top2vec` for unsupervised infererence of topics (analogous to parties here) in semantic space.

Interactive 2D/3D projections are made possible by exporting vectors and relevant metadata as separate 2D tensors to TensorBoard-compatible .TSV format (notebook-compatible visualisation utilities to be added, please refer to [this](https://github.com/mathiasbruun/DCPA) demo repo in the meantime).

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
│   └───tensorboard                 # Contains example of TensorBoard visualisation
│       │   tb_config.json
│       │   tb_doc_tensor.tsv
│       │   tb_metadata_tensor.tsv
│       │   tb_state.txt
│   
└───tests
│   │   politician2vec_test.ipynb   # Contains unit tests for politician2vec functions
│
└───politician2vec                  # Contains politican2vec functionality
│   │   Politician2Vec.py
│   │
│   └───utils
│       │   xxx.py
```

## Requirements

The current version of `politician2vec` was developed in Python 3.8.9. Demos will be provided as notebooks–for now, please refer to [this](https://github.com/mathiasbruun/DCPA) repo.

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

`$ pip install git+ssh://git@github.com/mathiasbruun/politician2vec.git`

## Example import

```python
from politician2vec import Politician2Vec
```

## Demos

Notebooks demonstrating common usecases and workflows to be added in `bin/notebooks/`.

Click <a href="https://projector.tensorflow.org/?config=https://bitbucket.org/advice-data-and-insights/tensorboard_input/raw/3a8dfc3fc19ef83d03832e89b207413ad918a1a3/projector_config.json" target="_blank">here</a> for TensorBoard visualisation example (works best in Google Chrome). When the embedding has loaded, check 'State 0' to load a pre-computed t-SNE projection—or switch to UMAP for a clearer distinction between politicians.

## Support

Please report any errors or suggestions to Mathias Bruun at [pvf607@alumni.ku.dk](mailto:pvf607@alumni.ku.dk).

## Foundational literature

- Angelov, D. (2020). *Top2Vec: Distributed Representations of Topics*. arXiv: [2008.09470](https://arxiv.org/abs/2008.09470).
- Bruun, M., A. Weile, A. Saabye & T. Blažková (n.d.). *'Splits' over splitting atoms in the EU*. Working paper.
- Carlsen, H. B. & S. Ralund (2022). Computational grounded theory revisited: From computer-led to computer-assisted text analysis. In: *Big Data & Society* 9.1 DOI: [10.1177/20539517221080146](http://journals.sagepub.com/doi/10.1177/20539517221080146).
- Le, Q. V. & T. Mikolov (2014). *Distributed Representations of Sentences and Documents*. arXiv: [1405.4053](https://arxiv.org/abs/1405.4053).
- Mikolov, T., K. Chen, G. Corrado & J. Dean (2013). *Efficient Estimation of Word Representations in Vector Space*. arXiv: [1301.3781](https://arxiv.org/abs/1301.3781).