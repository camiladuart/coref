#baseado no independent.py
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
        #removi:
        #self.genres: além do que preciso para as minhas tarefas
        #.subtoken_maps pq é da pipeline de dados -> modelo vai receber já os tensores prontos
        #.gold (gold labels - anotaçoes verdadeiras - o modelo recebe no max como entrada p calcular loss)
        #self.eval_data = None: dados de avaliação -> nao pertencem à classe do modelo

        #bert_config -> alterei para Transformers (nao preciso de um json separado). Encoder BERT:
        self.encoder = AutoModel.from_pretrained(config["encoder_name"]) # carrega modelo pronto (bert-base-cased) da biblio transformers
        hidden_size = self.encoder.config.hidden_size  #guarda o tamanho dos vetores - 768
        #removi: self.tokenizer

        # span_emb = [start ; end]. Cada span é representado juntando [vetor_start ; vetor_end]
        span_emb_size = hidden_size * 2 #ex: 768 no BERT base
        self.mention_scorer = nn.Linear(span_emb_size, 1) #cria uma camada linear que recebe esse vetor e devolve 1 numero só -> score 
            #score = quanto o modelo acha que aquele span é uma menção (numero alto = sim, baixo = não)



        #bloco de input_props.append do tensorflow antigo -> (pytorch) ao inves disso, vou definir as entradas diretamente nas funçoes:
        # input_ids [B,T], attention_mask [B,T], span_starts [N], span_ends [N], span_batch_idx [N]

        #placeholders e filas (PaddingFIFOQueue) -> removi para o pytorch (dataloader + chamar função)
        #removi o self.get_predictions_and_loss pq o trainer vai chamar

        #tvars, variable names é a inicialização de pesos via checkpoints TF -> removi pq o AutoModel.from_pretrained(config["encoder_name"]), ja carrega os pesos automaticamente (Hugging Face)
        #train/warmup/global steps e train_op

        '''funções que removi:
        1. start_enqueue_thread -> usa fila do tensorflow-> em PyTorch vira DataLoader no trainer
        2. restore -> restaura variáveis do TF a partir de um checkpoint-> aqui os pesos vem de AutoModel.from_pretrained(...)
        3. tensorize_mentions-> recebe lista de menções [(start, end), ...] e devolve dois arrays starts, ends: aqui o modelo vai já com span_starts/span_ends (tensores).
        4. tensorize_span_labels -> pega [(start, end, label_name), ...] e transforma em arrays + mapeia label_name ->id.
            necessária para classes além de 0/1; por enquanto tenho mention/not-mention (0/1) -> get_candidate_labels ja resolve isso por (start,end).
        5. get_speaker_dict- >cria um dicionário de speakers (não vou usar)
        6. tensorize_example -> TF + tokenizer dentro do modelo; aqui: tokenização por transformers.AutoTokenizer fora do modelo
        7. truncate_example-> corta o documento por número de sentenças e ajusta offsets de gold_starts/ends, sentence_map, etc.
            pode estar no DataLoader: em PyTorch, isso é feito antes de alimentar o modelo (se os exemplos ultrapassarem o limite)

        model sem leitura de arquivo, sem tokenizer, sem filas TF, sem restore, sem speaker/genre/cluster.
        '''


    #get_candidate_labels: alterei para implementação pytorch:
    def get_candidate_labels(
        self,
        candidate_starts: torch.LongTensor,  
        candidate_ends: torch.LongTensor,   
        labeled_starts: torch.LongTensor,  
        labeled_ends: torch.LongTensor      
    ) -> torch.LongTensor:                    
        device = candidate_starts.device #onde os tensores estão para devolver no mesmo lugar
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

