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
        self.head_attention = nn.Linear(hidden_size, 1)
        self.max_span_width = self.config.get("max_span_width", 30) #largura max
        self.span_width_emb_size = self.config.get("span_width_emb_size", 20) #dim do vetor de largura
        self.span_width_embeddings = nn.Embedding(self.max_span_width, self.span_width_emb_size) #cada largura vira um vetor
        
        # gêneros:
        self.use_genre = config.get("use_genre", False) #le do config
        if self.use_genre:
            self.genres = config["genres"] #pega do config a lista que eu defini
            self.genre_to_id = {g: i for i, g in enumerate(self.genres)} #cria os ids numericos
            self.genre_embeddings = nn.Embedding(
                num_embeddings=len(self.genres), #quantos generos
                embedding_dim=config["genre_emb_size"], #tamanho do vetor de cada genero
            )
        # span_emb = [start ; end ; head_attention]+width emb (+ opcionalmente gênero)
        span_emb_size = hidden_size * 3 + self.span_width_emb_size
        if self.use_genre:
            span_emb_size += config["genre_emb_size"]
            
        self.use_segment_distance = True
        self.max_training_sentences = config.get("max_training_sentences", 10)

        if self.use_segment_distance: #criando tabela de embeddings
            self.segment_distance_embeddings = nn.Embedding(
                self.max_training_sentences,
                span_emb_size
            )

        
        #projeção linear:
        self.span_projection = nn.Linear(span_emb_size, span_emb_size)
            #pega o vetor [start ; end] e aplica W*x + b (mantenho dimensão)
        
        self.mention_scorer = nn.Linear(span_emb_size, 1) #cria uma camada linear que recebe o vetor e devolve 1 numero só -> score 
            #score = quanto o modelo acha que aquele span é uma menção (numero alto = sim, baixo = não)

        # MLP para comparar pares de spans:
        #para cada par (i,j): [span_i ; span_j ; span_i * span_j]  ->  (3 * span_emb_size)
        pair_input_size = span_emb_size * 3
        if self.use_segment_distance:
            pair_input_size += span_emb_size
        self.pair_scorer = nn.Sequential(
            nn.Linear(pair_input_size, span_emb_size),
            nn.ReLU(),
            nn.Linear(span_emb_size, 1)  #devolve 1 score p par
        )
        self.last_pair_scores = None #guarda o último resultado de pares (visualização)

    #Head attention
    def compute_head_vecs(
        self,
        token_emb: torch.Tensor,
        span_starts: torch.LongTensor,
        span_ends: torch.LongTensor,
        span_batch_idx: torch.LongTensor,
    ) -> torch.Tensor:

        #1. determinar maximum span width 
        max_span_width = self.max_span_width
        #2. num de candidate spans
        num_spans = span_starts.size(0)
        #3. construir matriz de token indices para cada span
        offsets = torch.arange(max_span_width, device=span_starts.device).unsqueeze(0)  
        span_indices = span_starts.unsqueeze(1) + offsets  
        #4.evitar indexing errors:
        T_wp = token_emb.size(1)
        span_indices_clamped = span_indices.clamp(0, T_wp - 1)
        #5.juntar token embeddings para cada posiçao
        span_token_embs = token_emb[span_batch_idx.unsqueeze(1).expand_as(span_indices_clamped), span_indices_clamped]
        #6.criando mask: True para posicoes dentro do span, false para padding:
        span_mask = span_indices <= span_ends.unsqueeze(1) 
        #7.raw attention scores
        raw_scores = self.head_attention(span_token_embs).squeeze(-1) 
        #8.usando mask:
        raw_scores = raw_scores.masked_fill(~span_mask, -1e9)
        #9.softmax para pegar attention weights (cada coluna vai somar 1 nas posicoes validas)
        attn_weights = torch.softmax(raw_scores, dim=-1) 
        #10.calculo final: weighted sum:
        head_vecs = (attn_weights.unsqueeze(-1) * span_token_embs).sum(dim=1)

        return head_vecs

    def _forward_wp(
        self,
        input_ids: torch.LongTensor,       
        attention_mask: torch.LongTensor,  #coloquei tudo como entrada
        span_starts: torch.LongTensor,     
        span_ends: torch.LongTensor,       
        span_batch_idx: torch.LongTensor,
        span_segment_ids: torch.LongTensor, 
        candidate_cluster_ids: torch.LongTensor = None,
        mention_labels: torch.LongTensor = None,  
        genre=None,
        span_starts_tok=None, 
        span_ends_tok=None,
        return_debug: bool = False, 
    ) -> torch.Tensor:                    

        #pegar spans dos embeddings:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask) #transformação token -> embedding
        token_emb = outputs.last_hidden_state  #pegar os embeddings de cada token
        
        # gênero
        genre_emb = None
        if self.use_genre: 
            genre_emb = self.get_genre_embedding(genre, token_emb.device, token_emb.dtype)
            if genre_emb is None:
                # se não tiver genre ou não estiver na lista, usa vetor zero 
                genre_emb = torch.zeros(
                    self.config["genre_emb_size"],
                    device=token_emb.device,
                    dtype=token_emb.dtype,
                )

        
        #pegar vetor do token inicial e final + head attention
        start_vecs = token_emb[span_batch_idx, span_starts] 
        end_vecs   = token_emb[span_batch_idx, span_ends] 
        # width = número de tokens no span (inclui start e end)
        span_widths = span_ends - span_starts + 1                      
        span_widths = span_widths.clamp(1, self.max_span_width) #impede 0 e larguras acima do max
        span_width_ids = span_widths - 1                              
        width_vecs = self.span_width_embeddings(span_width_ids)        
        #chamo head attention:
        head_vecs = self.compute_head_vecs(
            token_emb=token_emb,
            span_starts=span_starts,
            span_ends=span_ends,
            span_batch_idx=span_batch_idx,
        )
        span_emb = torch.cat([start_vecs, end_vecs, head_vecs, width_vecs], dim=-1)  
        
        if self.use_genre:
            genre_feat = genre_emb.unsqueeze(0).expand(span_emb.size(0), -1) #replica o vetor do genero para todos os spans-todos pertencem ao mesmo doc
            span_emb = torch.cat([span_emb, genre_feat], dim=-1) #concatena ao embedding do span

        #proj linear:
        span_proj = self.span_projection(span_emb) 

        #beam:
        #scores de menção para todos os spans
        mention_scores = self.get_mention_scores(span_proj)
        #calculando k 
        num_words = input_ids.size(1) #aproximando pelo nº de tokens
        top_span_ratio = self.config.get("top_span_ratio", 0.4)
        max_k = 3900
        #no independent: k = min(3900, floor(num_words * top_span_ratio))
        k_float = float(num_words) * float(top_span_ratio)
        k = int(k_float)          
        k = min(max_k, k)
        #k não pode ser 0 nem maior que o num de spans
        N = mention_scores.size(0)
        if N == 0:
            return mention_scores.new_empty(0), mention_labels, mention_scores.new_tensor(0.0)
        k = max(1, min(k, N)) #nunca menor que 1 ou maior que N
        #calculando c:
        #no independent: c = min(max_top_antecedents, k)
        max_top_antecedents = self.config.get("max_top_antecedents", 50)
        c = min(max_top_antecedents, k)
        #pegando os k maiores scores
        top_scores, top_indices = torch.topk(mention_scores, k)
        
        #aplicar o beam: filtro -> só com spans do beam
        span_emb = span_emb[top_indices]
        span_starts = span_starts[top_indices]
        span_ends = span_ends[top_indices]
        mention_scores = mention_scores[top_indices]
        
        if span_starts_tok is not None and span_ends_tok is not None:
            span_starts_tok = span_starts_tok[top_indices]
            span_ends_tok   = span_ends_tok[top_indices]

        span_batch_idx = span_batch_idx[top_indices]
        
        if mention_labels is not None: #(se a menção for válida)
            mention_labels = mention_labels[top_indices]
        if candidate_cluster_ids is not None:
            candidate_cluster_ids = candidate_cluster_ids[top_indices]
        
        span_segment_ids = span_segment_ids[top_indices]
        self.last_pair_scores = self.score_span_pairs(span_emb, span_segment_ids, mention_scores)

        if return_debug:
            self.last_debug = {
                "top_scores": top_scores.detach().cpu(),
                "span_starts_tok": span_starts_tok.detach().cpu() if span_starts_tok is not None else None,
                "span_ends_tok": span_ends_tok.detach().cpu() if span_ends_tok is not None else None,
                "pair_scores": self.last_pair_scores.detach().cpu(),
            }
        else:
            self.last_debug = None

        # pega top-c antecedentes por span
        top_antecedents, top_antecedents_mask, top_antecedent_scores = \
            self.get_top_antecedents_and_scores(self.last_pair_scores, c)
        # dummy_scores = zeros([k, 1])-> inicio novo cluster (menção sem antecedente)
        dummy_scores = torch.zeros(
            (top_antecedent_scores.size(0), 1),
            device=top_antecedent_scores.device,
            dtype=top_antecedent_scores.dtype,
        )
        #equivalente a tf.concat([dummy_scores, top_antecedent_scores], 1) -> junta dummy com antec reais
        top_antecedent_scores = torch.cat([dummy_scores, top_antecedent_scores], dim=1) 

        
        #logits finais de menção -> scores dos spans do beam
        logits = top_scores
        
        device = token_emb.device
        dtype = mention_scores.dtype
        # zero "neutro" com dtype consistente
        loss = torch.zeros((), device=device, dtype=dtype)

        # loss de menção (span detection)
        mention_loss = torch.zeros((), device=device, dtype=dtype)
        if mention_labels is not None and mention_labels.numel() > 0:
            mention_loss = torch.nn.functional.binary_cross_entropy_with_logits( #função do pytorch
                logits,                    # = top_scores
                mention_labels.float()
            )

        loss = loss + mention_loss


        if candidate_cluster_ids is not None and candidate_cluster_ids.numel() > 0:
            loss = loss + self.coref_marginal_loss_with_dummy(
                self.last_pair_scores,
                candidate_cluster_ids.to(device),
            )

        return logits, mention_labels, loss
    
    def forward(
        self,
        *,
        sentences,                 
        seg_start: int,           
        seg_sents,                
        tokenizer,                
        gold_starts_all: torch.LongTensor,
        gold_ends_all: torch.LongTensor,
        gold_cluster_ids_all,
        max_span_width: int = 30,
        genre=None,
        span_segment_ids=None, return_debug: bool = False
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
            max_length=self.config.get("max_seq_length", 512),           # limite do BERT - ajustei o parametro no maintrainer
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
        span_starts_tok = span_starts.clone()
        span_ends_tok = span_ends.clone()

        # cada span candidato pertence ao segmento atual
        span_segment_ids = torch.full(
            (span_starts.size(0),),
            seg_start,
            dtype=torch.long,
            device=span_starts.device,
        )


        if span_starts.numel() == 0:
            return torch.empty(0, device=next(self.parameters()).device), torch.empty(0, dtype=torch.long, device=next(self.parameters()).device), torch.tensor(0.0, device=next(self.parameters()).device)

        # filtrar gold spans do segmento
        offset_glob = sum(len(s) for s in sentences[:seg_start])
        if gold_starts_all.numel() > 0:
            keep = (gold_starts_all >= offset_glob) & (gold_ends_all < offset_glob + len(tokens))
            gold_starts = gold_starts_all[keep] - offset_glob
            gold_ends   = gold_ends_all[keep]   - offset_glob
            gold_cluster_ids = gold_cluster_ids_all[keep]  
        else:
            empty = torch.empty(0, dtype=torch.long)
            gold_starts = empty
            gold_ends = empty
            gold_cluster_ids = empty  


        # wordpiece: converter spans tokens-> wp usando first_wp/last_wp
        keep_mask = []
        span_start_wp = []
        span_end_wp = []

        for s, e in zip(span_starts.tolist(), span_ends.tolist()):
            fs, le = first_wp[s], last_wp[e]
            ok = (0 <= fs <= le < T_wp)
            keep_mask.append(ok)
            if ok:
                span_start_wp.append(fs)
                span_end_wp.append(le)

        if len(span_start_wp) == 0:
            return torch.empty(0, device=next(self.parameters()).device), \
                torch.empty(0, dtype=torch.long, device=next(self.parameters()).device), \
                torch.tensor(0.0, device=next(self.parameters()).device)

        keep_mask = torch.tensor(keep_mask, dtype=torch.bool, device=span_starts.device)
        
        #rótulos 0/1: se o candidato coincide com algum gold (menção ou não)
        mention_labels = self.get_candidate_labels(
            span_starts, span_ends,
            gold_starts, gold_ends
        )
        #ids de cluster por candidato (0 = não pertence a nenhum cluster)
        candidate_cluster_ids = self.get_candidate_cluster_ids(
            span_starts,     
            span_ends,       
            gold_starts,    
            gold_ends,        
            gold_cluster_ids  
        )


        # filtra spans em token space e labels/cluster
        span_starts_tok = span_starts_tok[keep_mask]
        span_ends_tok   = span_ends_tok[keep_mask]
        mention_labels  = mention_labels[keep_mask]
        candidate_cluster_ids = candidate_cluster_ids[keep_mask]
        span_segment_ids = span_segment_ids[keep_mask]
        #substitui os spans por wp space (consertando erro anterior BERT)
        span_starts = torch.tensor(span_start_wp, dtype=torch.long, device=span_starts.device)
        span_ends   = torch.tensor(span_end_wp,   dtype=torch.long, device=span_ends.device)
        
        #para todos os tensores que entram em _forward_wp estarem no mesmo device:
        device = next(self.parameters()).device
        span_starts = span_starts.to(device)
        span_ends   = span_ends.to(device)
        span_segment_ids = span_segment_ids.to(device)
        span_batch_idx = torch.zeros_like(span_starts, device=device)

        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        mention_labels = mention_labels.to(device)
        candidate_cluster_ids = candidate_cluster_ids.to(device)

        #chama _forward_wp
        logits, mention_labels, loss = self._forward_wp(
            input_ids=input_ids,
            attention_mask=attention_mask,
            span_starts=span_starts,
            span_ends=span_ends,
            span_batch_idx=span_batch_idx,
            candidate_cluster_ids=candidate_cluster_ids, 
            mention_labels=mention_labels, 
            genre=genre,   
            span_segment_ids=span_segment_ids,
            span_starts_tok=span_starts_tok.to(device),
            span_ends_tok=span_ends_tok.to(device),
            return_debug=return_debug,
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

    def get_mention_scores(self, span_emb: torch.Tensor) -> torch.Tensor:
        return self.mention_scorer(span_emb).squeeze(-1)
    
    #calcular scores de relação entre pares de spans:
    def score_span_pairs(self, span_emb, span_segment_ids, mention_scores):
        N, D = span_emb.size()
        device = span_emb.device
        dtype = span_emb.dtype

        neg_inf = torch.tensor(-1e9, device=device, dtype=dtype)

        rows = []

        # i = 0 não tem antecedentes (tudo -1e9)
        rows.append(neg_inf.expand(N))

        for i in range(1, N):
            curr = span_emb[i].expand(i, D)   
            prev = span_emb[:i]             

            pair_feats = [curr, prev, curr * prev]

            mention_segment = span_segment_ids[i]
            antecedent_segment = span_segment_ids[:i]

            seg_dist = mention_segment - antecedent_segment
            seg_dist = torch.clamp(seg_dist, 0, self.max_training_sentences - 1)

            seg_emb = self.segment_distance_embeddings(seg_dist)  
            pair_feats.append(seg_emb)

            pair_input = torch.cat(pair_feats, dim=-1)            
            pairwise = self.pair_scorer(pair_input).squeeze(-1)     
            scores = pairwise + mention_scores[i] + mention_scores[:i] #score do span atual i e scores dos antecedentes

            # completa a linha com -1e9 para j >= i
            pad = neg_inf.expand(N - i)
            row = torch.cat([scores, pad], dim=0)                 
            rows.append(row)

        pair_scores = torch.stack(rows, dim=0)                   
        return pair_scores

    
    #para saber se dois spans candidatos pertencem ao mesmo cluster (estão ligados ou não):
    def get_candidate_cluster_ids( 
        self,
        candidate_starts: torch.LongTensor,   
        candidate_ends: torch.LongTensor,     
        gold_starts: torch.LongTensor,       
        gold_ends: torch.LongTensor,         
        gold_cluster_ids: torch.LongTensor,   
    ) -> torch.LongTensor:                    

        device = candidate_starts.device
        N = candidate_starts.size(0)

        if gold_starts.numel() == 0: #detectar casos vazios
            return torch.zeros(N, dtype=torch.long, device=device)

        #transformando candidatos e gold em matriz com start, end
        cand = torch.stack([candidate_starts, candidate_ends], dim=1)
        gold = torch.stack([gold_starts, gold_ends], dim=1)

        #comparando os candidatos com os gold: eq[i,j] = True se cand[i] == gold[j]
        eq = (cand[:, None, :] == gold[None, :, :]).all(dim=-1)  # [N, M]
        #transformando para boolean
        matched = eq.long()

        #multiplicação p pegar o cluster do span que bate com o gold (só 0 ou o id do cluster correto)
        cluster_ids = matched @ gold_cluster_ids.to(device)

        return cluster_ids
    
    #helper para transformar cluster_ids em matriz de rótulos e calcular loss ANTIGA - nao uso mais! -> substitui por coref_marginal_loss_with_dummy!!
    '''def coref_pair_loss(
        self,
        pair_scores: torch.Tensor,         
        candidate_cluster_ids: torch.Tensor 
    ) -> torch.Tensor:
        
        device = pair_scores.device
        N = candidate_cluster_ids.size(0)

        if N == 0:
            return torch.tensor(0.0, device=device)
        cid = candidate_cluster_ids

        #true para pares com i>j (antecedentes)
        tri_mask = torch.tril(torch.ones(N, N, dtype=torch.bool, device=device), diagonal=-1)
        
        #pruning de antecedentes:
        max_top_antecedents = self.config.get("max_top_antecedents", 50)
        c = min(max_top_antecedents, N) #numero maximo de antecedentes por span

        if c < N:
            masked_scores = pair_scores.masked_fill(~tri_mask, -1e9) #coloco num muito negativo onde nao for antecedente valido

            #guarda quais antecedentes manter p/ cada i
            keep_antecedent = torch.zeros_like(tri_mask)

            # para cada span i, manter só antecedentes com maior score
            for i in range(1, N):
                # scores só dos antecedentes válidos:
                row_scores = masked_scores[i, :i]   
                if row_scores.numel() == 0:
                    continue

                # número de antecedentes para manter nesta linha (não pode > i)
                k_i = min(c, i)

                # índices dos top-k_i antecedentes em j < i
                top_vals, top_idx = torch.topk(row_scores, k_i)

                # marcar esses antecedentes como "mantidos"
                keep_antecedent[i, top_idx] = True

            # combina: só pares i>j E escolhidos pelo top-c
            tri_mask = tri_mask & keep_antecedent


        # marco pares do mesmo cluster
        same_cluster = (cid[:, None] == cid[None, :]) & (cid[:, None] != 0)

        # 1 se são do mesmo cluster e j é antecedente de i:
        pair_labels = (same_cluster & tri_mask).float()  

        # scores e rótulos só dos pares considerados (pruning de c já aplicado)
        valid_scores = pair_scores[tri_mask]
        valid_labels = pair_labels[tri_mask]

        if valid_scores.numel() == 0:
            return torch.tensor(0.0, device=device)

        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            valid_scores, valid_labels
        )
        return loss '''
    
    def get_genre_embedding(self, genre, device, dtype): #genero para embedding
        if not self.use_genre or genre is None:
            return None

        if isinstance(genre, str):
            genre = genre.strip().lower() #normaliza
            genre_id = self.genre_to_id.get(genre, None) #converte string -> id
            if genre_id is None:
                return None
        else:
            genre_id = int(genre)

        genre_id = torch.tensor([genre_id], device=device)
        emb = self.genre_embeddings(genre_id).squeeze(0)
        return emb.to(dtype=dtype)            
    
    #calculo da loss com dummy antecedent (somada à loss de menção na forward)
    def coref_marginal_loss_with_dummy(
        self,
        pair_scores: torch.Tensor,          
        candidate_cluster_ids: torch.Tensor 
    ) -> torch.Tensor:
        device = pair_scores.device
        dtype = pair_scores.dtype
        k = candidate_cluster_ids.size(0)

        if k == 0:
            return torch.tensor(0.0, device=device, dtype=dtype)

        c = min(self.config.get("max_top_antecedents", 50), k)

        total = torch.tensor(0.0, device=device, dtype=dtype)

        for i in range(k):
            # dummy score sempre 0.0
            dummy = torch.zeros(1, device=device, dtype=dtype)

            if i == 0:
                # só dummy
                log_norm = torch.logsumexp(dummy, dim=0)
                log_gold = log_norm
                total = total + (log_norm - log_gold)
                continue

            row = pair_scores[i, :i]              
            ci = min(c, i)

            top_vals, top_idx = torch.topk(row, k=ci)  

            # distribuição: [dummy + top_vals]
            scores = torch.cat([dummy, top_vals], dim=0)  
            log_norm = torch.logsumexp(scores, dim=0)

            cid_i = candidate_cluster_ids[i]
            if cid_i != 0:
                ant_cids = candidate_cluster_ids[top_idx]   
                correct = (ant_cids == cid_i)               
                if correct.any():
                    gold_scores = top_vals[correct]         
                    log_gold = torch.logsumexp(gold_scores, dim=0)
                else:
                    # gold = dummy
                    log_gold = dummy.squeeze(0)
            else:
                log_gold = dummy.squeeze(0)

            total = total + (log_norm - log_gold)

        return total

    
    def get_top_antecedents_and_scores(self, pair_scores: torch.Tensor, c: int):
        device = pair_scores.device
        k = pair_scores.size(0)

        top_antecedents = torch.full((k, c), -1, dtype=torch.long, device=device)
        top_scores = torch.full((k, c), -1e9, dtype=pair_scores.dtype, device=device)
        top_mask = torch.zeros((k, c), dtype=torch.bool, device=device)

        for i in range(k):
            if i == 0:
                continue
            ci = min(c, i)
            vals, idx = torch.topk(pair_scores[i, :i], k=ci)
            top_antecedents[i, :ci] = idx
            top_scores[i, :ci] = vals
            top_mask[i, :ci] = True

        return top_antecedents, top_mask, top_scores