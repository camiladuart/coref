#baseado no independent.py
import torch
import torch.nn as nn
from transformers import AutoModel

class CorefModel(nn.Module): #nn.Module do torch.nn -> lidar com classes com camadas e cálculos
    def __init__(self, config): 
        super().__init__()
        self.config = config #config dicionario com hiperparametros
       
        #bert_config -> alterei para Transformers. Encoder BERT:
        self.encoder = AutoModel.from_pretrained(config["encoder_name"]) # carrega modelo pronto (bert-base-cased)
        hidden_size = self.encoder.config.hidden_size  #guarda o tamanho dos vetores - 768

        # span_emb = [start ; end]. Cada span é representado juntando [vetor_start ; vetor_end]
        span_emb_size = hidden_size * 2 #ex: 768 no BERT base
        self.mention_scorer = nn.Linear(span_emb_size, 1) #cria uma camada linear que recebe o vetor e devolve 1 numero só -> score 
            #score = quanto o modelo acha que aquele span é uma menção (numero alto = sim, baixo = não)
      

    def forward(
        self,
        input_ids: torch.LongTensor,       
        attention_mask: torch.LongTensor,  #coloquei tudo como entrada
        span_starts: torch.LongTensor,     
        span_ends: torch.LongTensor,       
        span_batch_idx: torch.LongTensor  
    ) -> torch.Tensor:                    

        #pegar spans dos embeddings:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask) #transformação token -> embedding
        token_emb = outputs.last_hidden_state  #pegar os embeddings de cada token
        #pegar vetor do token inicial e final 
        start_vecs = token_emb[span_batch_idx, span_starts] 
        end_vecs   = token_emb[span_batch_idx, span_ends]   
        #juntar os dois vetores em um só
        span_emb = torch.cat([start_vecs, end_vecs], dim=-1) 
         
        #calcular o score da menção (n alto-> prov menção; baixo-> nao é):
        logits = self.get_mention_scores(span_emb)
        return logits #score


    #get_candidate_labels: alterei para implementação pytorch
    def get_candidate_labels(
        self,
        candidate_starts: torch.LongTensor,  
        candidate_ends: torch.LongTensor,   
        labeled_starts: torch.LongTensor,  
        labeled_ends: torch.LongTensor      
    ) -> torch.LongTensor:                    
        device = candidate_starts.device #device: onde os tensores estão para devolver no mesmo lugar
        N = candidate_starts.size(0) #n = quantidade de candidatos
        #recebe os inícios e finais dos candidatos e dos gold (mençoes verdadeiras)

        #caso lista vazia (0 spans gold): 
        if labeled_starts.numel() == 0:
            return torch.zeros(N, dtype=torch.long, device=device)
            
        #juntando início e fim de cada span em pares:
        cand = torch.stack([candidate_starts, candidate_ends], dim=1)  # cand: matriz [N,2]: cada linha: start e end de um candidato
        gold = torch.stack([labeled_starts, labeled_ends], dim=1)      # gold: [M,2] cada linha: start, end de um gold
        eq = (cand[:, None, :] == gold[None, :, :]).all(dim=-1)        # [N,M] compara todos os candidatos com todos os gold: eq[i, j] = True se o candidato i é igual ao gold j
        return eq.any(dim=1).long()                                     #p cada candidato i, verifica se ele bate com algum gold (linha i tem algum True?)
        #resultado final é um rotulo por candidato

    
    #tensorflow -> transformers
    def get_prediction_and_loss(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        span_starts: torch.LongTensor,
        span_ends: torch.LongTensor,
        span_batch_idx: torch.LongTensor,
        mention_labels: torch.LongTensor = None
):
        #padronizando a máscara
        attention_mask = attention_mask.long()

        logits = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            span_starts=span_starts,
            span_ends=span_ends,
            span_batch_idx=span_batch_idx
        )

        #loss=erro (compara verdadeiro com as predictions feitas)
        loss = None #começa com loss vazia (se não houver rótulos, nao precisa calcular nada)
        if mention_labels is not None:
            if mention_labels.numel() == 0:
                loss = logits.new_tensor(0.0)
            else:
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, mention_labels.float()
                ) #binary_cross_entropy_with_logits para comparar logits (notas brutas que o modelo deu) com mention_labels (rótulos verdadeiros) 
                    #e mede o quanto o modelo errou
        return {"logits": logits, "loss": loss}
    
    def get_mention_scores(self, span_emb: torch.Tensor) -> torch.Tensor:
        return self.mention_scorer(span_emb).squeeze(-1)