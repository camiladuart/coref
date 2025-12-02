#baseado no independent.py
import torch
import torch.nn as nn
from transformers import AutoModel
from coref_loader.data import flatten_sentences, build_candidates

class CorefModel(nn.Module): #nn.Module do torch.nn -> lidar com classes com camadas e cálculos
    def __init__(self, config): 
        super().__init__()
        self.config = config #config dicionario com hiperparametros
       
        #bert_config -> alterei para Transformers. Encoder BERT:
        self.encoder = AutoModel.from_pretrained(config["encoder_name"]) # carrega modelo pronto (bert-base-cased)
        hidden_size = self.encoder.config.hidden_size  #guarda o tamanho dos vetores - 768

        # span_emb = [start ; end]. Cada span é representado juntando [vetor_start ; vetor_end]
        span_emb_size = hidden_size * 2 #ex: 768 no BERT base
        
        #projeção linear:
        self.span_projection = nn.Linear(span_emb_size, span_emb_size)
            #pega o vetor [start ; end] e aplica W*x + b (mantenho dimensão)
        
        self.mention_scorer = nn.Linear(span_emb_size, 1) #cria uma camada linear que recebe o vetor e devolve 1 numero só -> score 
            #score = quanto o modelo acha que aquele span é uma menção (numero alto = sim, baixo = não)

        # MLP para comparar pares de spans:
        #para cada par (i,j): [span_i ; span_j ; span_i * span_j]  ->  (3 * span_emb_size)
        pair_input_size = span_emb_size * 3
        self.pair_scorer = nn.Sequential(
            nn.Linear(pair_input_size, span_emb_size),
            nn.ReLU(),
            nn.Linear(span_emb_size, 1)  #devolve 1 score p par
        )
        self.last_pair_scores = None #guarda o último resultado de pares (visualização)


    def _forward_wp(
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
        #proj linear:
        span_proj = self.span_projection(span_emb) 
        
        #calculando scores de relação entre spans:
        self.last_pair_scores = self.score_span_pairs(span_emb)

        #calculando o score da menção (n alto-> prov menção; baixo-> nao é):
        logits = self.get_mention_scores(span_proj)
        
        return logits #score
    
    def forward(
        self,
        *,
        sentences,                 
        seg_start: int,           
        seg_sents,                
        tokenizer,                
        gold_starts_all: torch.LongTensor,
        gold_ends_all: torch.LongTensor,
        max_span_width: int = 30,
    ):  
    # juntar sentenças e construir sentence_map
        tokens, sentence_map = flatten_sentences(seg_sents)
        if not tokens:
            # segmento vazio
            return torch.empty(0, device=next(self.parameters()).device), torch.empty(0, dtype=torch.long, device=next(self.parameters()).device), torch.tensor(0.0, device=next(self.parameters()).device)

        # tokenizar: transf tokens em ids numéricos p/ encoder (com truncagem para 512) + criação da mask
        enc = tokenizer(
            tokens,
            is_split_into_words=True,
            add_special_tokens=False,
            truncation=True,          # truncagem agora é por segmento (até 512 WPs), não no doc inteiro
            max_length=512,           # limite do BERT
            return_tensors="pt"
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        T_wp = input_ids.size(1)  # quantos subtokens tem este segmento: n são tokens pq o bert quebra palavras em pedacinhos
        
        #wordpiece
        try: #p garantir compatibilidade com versoes do tokenizer
            wp2tok = enc.word_ids(0)
        except Exception:
            wp2tok = enc.encodings[0].word_ids
            
        #construir primeiro/último WP de cada token 
        n_tokens = len(tokens)
        first_wp = [-1] * n_tokens #listas começam com -1
        last_wp = [-1] * n_tokens
        for wp_idx in range(T_wp):     #percorrer todos os wordpieces do segmento    
            tok_idx = wp2tok[wp_idx] #dizer qual token gerou esse wordpiece
            if tok_idx is None:
                continue
            if first_wp[tok_idx] == -1:     # se for a primeira vez vendo o tok_idx, guarda em first
                first_wp[tok_idx] = wp_idx
            last_wp[tok_idx] = wp_idx #sempre atualizo no final (utlimo wp que vi para esse token)
    
        # gerando candidatos (que não cruzam sentença, largura <= max_span_width)
        span_starts, span_ends = build_candidates(sentence_map, max_span_width)
        if span_starts.numel() == 0:
            return torch.empty(0, device=next(self.parameters()).device), torch.empty(0, dtype=torch.long, device=next(self.parameters()).device), torch.tensor(0.0, device=next(self.parameters()).device)

        # filtrar gold spans do segmento
        offset_glob = sum(len(s) for s in sentences[:seg_start])#inicio do segmento (soma quantos tokens existem antes do segmento)
        if gold_starts_all.numel() > 0: #segue se ha gold spans
            keep = (gold_starts_all >= offset_glob) & (gold_ends_all < offset_glob + len(tokens)) #cria uma mascara booleana (keep) pra pegar só as mençoes que caem dentro do inertavalo desse segmento (gold_starts_all e gold_ends_all)
            gold_starts = gold_starts_all[keep] - offset_glob
            gold_ends = gold_ends_all[keep] - offset_glob
        else:
            gold_starts = gold_ends = torch.empty(0, dtype=torch.long)

        mention_labels = self.get_candidate_labels(span_starts, span_ends, gold_starts, gold_ends)

        # wordpiece: converter spans tokens-> wp usando first_wp/last_wp
        span_start_wp, span_end_wp = [], []
        for s, e in zip(span_starts.tolist(), span_ends.tolist()):
            fs, le = first_wp[s], last_wp[e]
            if 0 <= fs <= le < T_wp:
                span_start_wp.append(fs)
                span_end_wp.append(le)
        if not span_start_wp:
            return torch.empty(0, device=next(self.parameters()).device), torch.empty(0, dtype=torch.long, device=next(self.parameters()).device), torch.tensor(0.0, device=next(self.parameters()).device)

        #para todos os tensores que entram em _forward_wp estarem no mesmo device:
        device = next(self.parameters()).device
        span_starts = span_starts.to(device)
        span_ends   = span_ends.to(device)
        span_batch_idx = torch.zeros_like(span_starts, device=device)

        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        mention_labels = mention_labels.to(device)


        #chama _forward_wp
        logits = self._forward_wp(
            input_ids=input_ids,
            attention_mask=attention_mask,
            span_starts=span_starts,
            span_ends=span_ends,
            span_batch_idx=span_batch_idx,
        )

        # calculando loss dentro da forward
        if mention_labels.numel() == 0:
            loss = logits.new_tensor(0.0)
        else:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, mention_labels.float()
            )

        return logits, mention_labels.to(device), loss

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
    # Versão antiga get_prediction_and_loss para entrada já em tensores WP.
    #nao chamo porque faço o pré-processamento dentro do forward -> loss está sendo calculada direto no trainer
    '''
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
    '''
    def get_mention_scores(self, span_emb: torch.Tensor) -> torch.Tensor:
        return self.mention_scorer(span_emb).squeeze(-1)
    
    #calcular scores de relação entre pares de spans:
    def score_span_pairs(self, span_emb: torch.Tensor) -> torch.Tensor:
        N, D = span_emb.size() # N numero de spans; dimensão D
        device = span_emb.device

        #inicializa com valor bem negativo (p someçar "sem relaçao")
        pair_scores = span_emb.new_full((N, N), fill_value=-1e9)

        #para cada span i, comparamos com spans antecedentes j < i 
        for i in range(1, N):
            curr = span_emb[i].expand(i, D)     
            prev = span_emb[:i]                 #spans 0..i-1  -> [i, D]

            #representação do par (i,j): [span_i ; span_j ; span_i * span_j]
            pair_input = torch.cat(
                [curr, prev, curr * prev],
                dim=-1
            )                                   # [i, 3D]

            #passando no MLP de pares para gerar um score por par:
            scores = self.pair_scorer(pair_input).squeeze(-1)  #squeeze(-1) p transformar de [i, 1] matriz 2D para [i], vetor 1D
            pair_scores[i, :i] = scores #guardando
            
        return pair_scores