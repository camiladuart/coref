# Coreference Resolution

Code used for the MSc thesis **“Coreference in Context: Portuguese Evaluation of Multilingual and Monolingual Models”**.

This repository contains the data preprocessing, training, prediction, and evaluation code used for the coreference resolution experiments in the thesis. The implementation follows a span-ranking approach with pretrained Transformer encoders.

The experiments include language-specific models and a multilingual model trained on Romance-language datasets. Coreference performance is evaluated with MUC, B³, CEAF-e, and CoNLL F1.

## Implementation

The coreference resolution implementation in this repository is based on the
BERT-based coreference resolution implementation by Joshi et al. (2019):

https://github.com/mandarjoshi90/coref

The original implementation is distributed under the Apache License 2.0.
The code was adapted in this work for the datasets, pretrained encoders,
training configurations, prediction pipeline, and evaluation procedure used
in the thesis.

## Multilingual model

The selected checkpoints of the Romance multilingual model used in the final evaluation are available on Hugging Face:

https://huggingface.co/camilaalves/romance-coreference-mdeberta-v3-base

## Thesis

Camila Alves. *Coreference in Context: Portuguese Evaluation of Multilingual and Monolingual Models*. MSc thesis, University of Porto, 2026.
