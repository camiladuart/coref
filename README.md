# Coreference Resolution

Code used for the MSc thesis **“Coreference in Context: Portuguese Evaluation of Multilingual and Monolingual Models”**.

This repository contains the data preprocessing, training, prediction, and evaluation code used for the coreference resolution experiments in the thesis. The implementation follows a span-ranking approach with pretrained Transformer encoders.

The experiments include language-specific models and a multilingual model trained on Romance-language datasets. Coreference performance is evaluated with MUC, B³, CEAF-e, and CoNLL F1.

## Multilingual model

The selected checkpoints of the Romance multilingual model used in the final evaluation are available on Hugging Face:

https://huggingface.co/camilaalves/romance-coreference-mdeberta-v3-base

## Thesis

Camila Alves. *Coreference in Context: Portuguese Evaluation of Multilingual and Monolingual Models*. MSc thesis, University of Porto, 2026.
