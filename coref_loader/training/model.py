import os
import random
import json
import threading
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class CorefModel(nn.Module): #nn.Modulo do torch.nn -> lidar com classes com camadas e cálculos
    def __init__(self, config): #construtor:
        super().__init__() #chamar o inicializador da nn.Module
        # 1) hiperparâmetros principais
        self.config = config #config dicionario com hiperparametros
        self.max_segment_len = config.get("max_segment_len", 512) #dividir doc em segmentos (BERT limite 512 tokens)
        self.max_span_width = config.get("max_span_width", 30) #importante p limitar o tamanho de cada candidato

        # 2) encoder de linguagem
        self.encoder = AutoModel.from_pretrained(config["encoder_name"])
        hidden_size = self.encoder.config.hidden_size  # ex.: 768 para BERT base

        # 3) dimensão da representação de span: start + end + width_emb
        span_width_emb_size = config.get("span_width_emb_size", 20)
        self.width_embeddings = nn.Embedding(self.max_span_width + 1, span_width_emb_size)

        # 4) projetor para o embedding final do span (se for preciso ajustes de dimensão)
        span_emb_size = hidden_size * 2 + span_width_emb_size  # [start ; end ; width_emb]
        self.span_projector = nn.Sequential(
            nn.Linear(span_emb_size, span_emb_size),
            nn.ReLU(),
            nn.Dropout(config.get("dropout", 0.2)),
        )

        # 5) scorer de menções (get_mention_scores): um MLP que produz 1 logit por span
        self.mention_scorer = nn.Sequential(
            nn.Linear(span_emb_size, span_emb_size // 2),
            nn.ReLU(),
            nn.Dropout(config.get("dropout", 0.2)),
            nn.Linear(span_emb_size // 2, 1)  # logit
        )
