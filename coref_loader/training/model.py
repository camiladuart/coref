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
        #add droupout: 
        self.dropout_rate = self.config.get("dropout_rate", 0.3)
        self.dropout = nn.Dropout(self.dropout_rate)
        
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
        self.max_training_sentences = config.get("max_training_sentences", 50)

        if self.use_segment_distance:
            self.seg_dist_emb_size = 20
            self.segment_distance_embeddings = nn.Embedding(
                self.max_training_sentences,
                self.seg_dist_emb_size
            )

        #speakers info:
        self.use_speakers = config.get("use_speakers", False)
        if self.use_speakers:
            self.speaker_emb_size = config.get("speaker_emb_size", 20)
            # 3 casos: mesmo speaker, speakers diferentes, speaker desconhecido
            self.speaker_embeddings = nn.Embedding(3, self.speaker_emb_size)
        #ajustando pair_input_size:
        pair_input_size = span_emb_size * 3
        if self.use_segment_distance:
            pair_input_size += self.seg_dist_emb_size
        if self.use_speakers:
            pair_input_size += self.speaker_emb_size
        
        #score = quanto o modelo acha que aquele span é uma menção (numero alto = sim, baixo = não)
        #Deepen mention scorer para FFNN com 2 hidden layers + dropout
        ffnn_size = self.config.get("ffnn_size", span_emb_size)
        self.mention_scorer = nn.Sequential(
            nn.Linear(span_emb_size, ffnn_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(ffnn_size, ffnn_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(ffnn_size, 1),
        )

        #para comparar pares de spans:
        pair_ffnn_size = self.config.get("pair_ffnn_size", span_emb_size)
        self.pair_scorer = nn.Sequential(
            nn.Linear(pair_input_size, pair_ffnn_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(pair_ffnn_size, pair_ffnn_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(pair_ffnn_size, 1),
        )
        self.last_pair_scores = None #guarda o último resultado de pares 

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
    
    #normalizando speakers para usar speakers info sem ter problemas em formato de datasets:
    def normalize_speakers(self, speakers, sentences): #Retorna speakers no formato List[List[str]] com mesmo shape de sentences, ou None se não der para usar speakers com segurança
        if speakers is None:
            return None

        if sentences is None or not isinstance(sentences, list):
            return None

        #se já veio como lista de listas (ideal):
        if (
            isinstance(speakers, list)
            and len(speakers) == len(sentences)
            and all(isinstance(x, list) for x in speakers)
        ):
            normalized = []
            for sent_tokens, sent_spk in zip(sentences, speakers):
                if not isinstance(sent_tokens, list):
                    return None
                if sent_spk is None:
                    normalized.append(["UNK"] * len(sent_tokens))
                    continue
                if not isinstance(sent_spk, list):
                    return None

                #se:mesmo tamanho da sentença 
                if len(sent_spk) == len(sent_tokens):
                    normalized.append([str(s) if s is not None else "UNK" for s in sent_spk])
                #1 speaker para a sentença inteira -> replica
                elif len(sent_spk) == 1:
                    spk = str(sent_spk[0]) if sent_spk[0] is not None else "UNK"
                    normalized.append([spk] * len(sent_tokens))
                else:
                    #formato inconsistente
                    return None
            return normalized

        #se speakers é uma lista plana do documento inteiro:
        if isinstance(speakers, list) and all(not isinstance(x, list) for x in speakers):
            flat_tokens = [tok for sent in sentences for tok in sent]
            if len(speakers) == len(flat_tokens):
                normalized = []
                pos = 0
                for sent_tokens in sentences:
                    n = len(sent_tokens)
                    chunk = speakers[pos:pos+n]
                    normalized.append([str(s) if s is not None else "UNK" for s in chunk])
                    pos += n
                return normalized

        #se for 1 speaker por sentença:
        if isinstance(speakers, list) and len(speakers) == len(sentences):
            if all(not isinstance(x, list) for x in speakers):
                normalized = []
                for sent_tokens, spk in zip(sentences, speakers):
                    speaker_name = str(spk) if spk is not None else "UNK"
                    normalized.append([speaker_name] * len(sent_tokens))
                return normalized

        #se for qualquer outro formato: desliga speakers para esse documento
        return None

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
        speaker_ids=None,
        return_debug: bool = False, 
    ) -> torch.Tensor:                    

        #pegar spans dos embeddings:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask) #transformação token -> embedding
        token_emb = outputs.last_hidden_state  #pegar os embeddings de cada token
        token_emb = self.dropout(token_emb)
        
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

        span_emb = self.dropout(span_emb)
        
        #beam:
        #scores de menção para todos os spans
        mention_scores = self.get_mention_scores(span_emb)
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

        #ordenando beam por posição no texto (start crescente, depois end crescente) - alterei
        T = span_ends[top_indices].max().item() + 1  # nº real de tokens no segmento
        sort_order = torch.argsort(
            span_starts[top_indices] * T + span_ends[top_indices]
        )
        top_indices = top_indices[sort_order]
        top_scores  = top_scores[sort_order]

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
        #filtrando speaker_ids com o beam
        beam_speaker_ids = None
        if speaker_ids is not None:
            beam_speaker_ids = speaker_ids[top_indices]
            
        self.last_pair_scores = self.score_span_pairs(
            span_emb, span_segment_ids, mention_scores,
            speaker_ids=beam_speaker_ids
        )

        if return_debug:
            self.last_debug = {
                "top_scores": top_scores.detach().cpu(),
                "span_starts_tok": span_starts_tok.detach().cpu() if span_starts_tok is not None else None,
                "span_ends_tok": span_ends_tok.detach().cpu() if span_ends_tok is not None else None,
                "pair_scores": self.last_pair_scores.detach().cpu(),
                "span_emb": span_emb.detach(),
                "mention_scores": mention_scores.detach(),
                "span_segment_ids": span_segment_ids.detach(),
            }
        else:
            self.last_debug = None

        # top-c antecedentes por span
        top_ant_scores, top_ant_idx = torch.topk(self.last_pair_scores, k=c, dim=1)       
        
        #logits finais de menção -> scores dos spans do beam
        logits = top_scores
        
        device = token_emb.device
        dtype = mention_scores.dtype
        # zero "neutro" com dtype consistente
        loss = torch.zeros((), device=device, dtype=dtype)

        # loss de menção (span detection)
        mention_loss = torch.zeros((), device=device, dtype=dtype)
        if mention_labels is not None and mention_labels.numel() > 0:
            n_pos = mention_labels.sum().float().clamp(min=1.0)
            n_neg = (mention_labels == 0).sum().float().clamp(min=1.0)
            pos_weight = (n_neg / n_pos).clamp(max=20.0)
            mention_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits,
                mention_labels.float(),
                pos_weight=pos_weight,
            )

        loss = loss + mention_loss


        if candidate_cluster_ids is not None and candidate_cluster_ids.numel() > 0:
            loss = loss + self.coref_marginal_loss_with_dummy(
                top_ant_scores,                
                top_ant_idx,                    
                candidate_cluster_ids.to(device),
            )

        return logits, mention_labels, loss
    
    #Implementação do pooling: Divide o documento em segmentos de <= max_seq_length wordpieces (nunca trunca),
    #codifica cada segmento, e corre uma etapa de antecedente global sobre todos os spans do documento:
    def forward_document(
        self,
        *,
        sentences,
        tokenizer,
        gold_starts_all: torch.Tensor,
        gold_ends_all: torch.Tensor,
        gold_cluster_ids_all,
        max_span_width: int = 30,
        genre=None,
        speakers=None,
        return_debug: bool = False,
    ):
        
        device = next(self.parameters()).device
        max_seq_len = self.config.get("max_seq_length", 512)

        #Passo 1: dividir documento em segmentos por orçamento de WPs, nao por número de frases — evita truncagem)
        segments = []          # cada entry: (sent_start_idx, sent_end_idx_excl, tokens, wp_ids)
        current_tokens = []
        current_sent_start = 0
        current_sent_end = 0

        for sent_idx, sent in enumerate(sentences):
            # tentativa: adicionar esta frase ao segmento atual
            candidate = current_tokens + sent
            enc_test = tokenizer(
                candidate,
                is_split_into_words=True,
                add_special_tokens=False,
                truncation=False,
                return_tensors="pt",
            )
            if enc_test["input_ids"].size(1) > max_seq_len and current_tokens:
                # esta frase não cabe — fechar segmento atual e começar novo
                segments.append((current_sent_start, current_sent_end, current_tokens))
                current_tokens = sent
                current_sent_start = sent_idx
                current_sent_end = sent_idx + 1
            else:
                current_tokens = candidate
                current_sent_end = sent_idx + 1

        if current_tokens:
            segments.append((current_sent_start, current_sent_end, current_tokens))

        if not segments:
            empty = torch.empty(0, device=device)
            return empty, empty.long(), torch.tensor(0.0, device=device)

        #Passo 2: codificar cada segmento e recolher embeddings de token
        all_token_emb = []   
        token_offset = 0     

        for seg_sent_start, seg_sent_end, seg_tokens in segments:
            enc = tokenizer(
                seg_tokens,
                is_split_into_words=True,
                add_special_tokens=False,
                truncation=False,   # NUNCA trunca — o orçamento já garantiu que cabe
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            T_wp = input_ids.size(1)

            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            token_emb_wp = self.dropout(outputs.last_hidden_state.squeeze(0))  # [T_wp, H]

            # mapear WP -> token: pegar o primeiro WP de cada token
            try:
                wp2tok = enc.word_ids(0)
            except Exception:
                wp2tok = enc.encodings[0].word_ids

            n_tok = len(seg_tokens)
            first_wp = [-1] * n_tok
            last_wp  = [-1] * n_tok
            for wp_i in range(T_wp):
                tok_i = wp2tok[wp_i]
                if tok_i is None:
                    continue
                if first_wp[tok_i] == -1:
                    first_wp[tok_i] = wp_i
                last_wp[tok_i] = wp_i

            # representação por token = embedding do primeiro WP do token
            tok_embs = []
            for t in range(n_tok):
                if first_wp[t] != -1:
                    tok_embs.append(token_emb_wp[first_wp[t]])
                else:
                    tok_embs.append(torch.zeros(token_emb_wp.size(-1), device=device, dtype=token_emb_wp.dtype))
            tok_embs = torch.stack(tok_embs, dim=0)  

            all_token_emb.append(tok_embs)

            # guardar first_wp/last_wp para uso na construção de spans
            segments[len(all_token_emb) - 1] = (seg_sent_start, seg_sent_end, seg_tokens, first_wp, last_wp, token_offset, T_wp, enc)

            token_offset += n_tok

        all_token_emb_cat = torch.cat(all_token_emb, dim=0)  

        #Passo 3: construir candidatos a span ao nível do documento
        all_span_starts = []
        all_span_ends   = []
        all_span_segment_ids = []
        all_span_seg_idx = []   # índice numérico do segmento (0, 1, 2, ...)

        doc_offset = 0  # offset de token dentro do documento inteiro

        for seg_idx, seg_data in enumerate(segments):
            seg_sent_start, seg_sent_end, seg_tokens, first_wp, last_wp, tok_off, T_wp, enc = seg_data
            seg_sents = sentences[seg_sent_start:seg_sent_end]

            # sentence_map para este segmento
            from coref_loader.data import flatten_sentences, build_candidates
            _, sentence_map = flatten_sentences(seg_sents)
            span_s, span_e = build_candidates(sentence_map, max_span_width)

            # filtrar spans cujos WPs estão dentro do segmento
            for s, e in zip(span_s.tolist(), span_e.tolist()):
                fs, le = first_wp[s], last_wp[e]
                if 0 <= fs <= le < T_wp:
                    all_span_starts.append(s + doc_offset)
                    all_span_ends.append(e + doc_offset)
                    all_span_segment_ids.append(seg_sent_start)
                    all_span_seg_idx.append(seg_idx)

            doc_offset += len(seg_tokens)

        if not all_span_starts:
            empty = torch.empty(0, device=device)
            return empty, empty.long(), torch.tensor(0.0, device=device)

        span_starts_tok  = torch.tensor(all_span_starts,     dtype=torch.long, device=device)
        span_ends_tok    = torch.tensor(all_span_ends,       dtype=torch.long, device=device)
        span_segment_ids = torch.tensor(all_span_segment_ids, dtype=torch.long, device=device)
        all_span_segment_ids_raw = torch.tensor(all_span_seg_idx, dtype=torch.long, device=device)

        #Passo 4: construir span embeddings 
        N_spans = span_starts_tok.size(0)

        start_vecs = all_token_emb_cat[span_starts_tok]
        end_vecs   = all_token_emb_cat[span_ends_tok]

        span_widths = (span_ends_tok - span_starts_tok + 1).clamp(1, self.max_span_width)
        width_vecs  = self.span_width_embeddings(span_widths - 1)

        # head attention usando all_token_emb_cat
        max_w = self.max_span_width
        offsets = torch.arange(max_w, device=device).unsqueeze(0)
        span_indices = span_starts_tok.unsqueeze(1) + offsets
        span_indices_clamped = span_indices.clamp(0, all_token_emb_cat.size(0) - 1)
        span_token_embs = all_token_emb_cat[span_indices_clamped]

        # cortar janela de atenção no limite do segmento do span
        seg_end_tok = torch.zeros(span_starts_tok.size(0), dtype=torch.long, device=device)
        for seg_idx, seg_data in enumerate(segments):
            seg_sent_start, seg_sent_end, seg_tokens, first_wp, last_wp, tok_off, T_wp, enc = seg_data
            seg_last_tok = tok_off + len(seg_tokens) - 1
            belongs = (all_span_segment_ids_raw == seg_idx)
            seg_end_tok[belongs] = seg_last_tok

        span_mask = (span_indices <= span_ends_tok.unsqueeze(1)) & \
                    (span_indices <= seg_end_tok.unsqueeze(1))
        raw_scores = self.head_attention(span_token_embs).squeeze(-1)
        raw_scores = raw_scores.masked_fill(~span_mask, -1e9)
        attn_weights = torch.softmax(raw_scores, dim=-1)
        head_vecs = (attn_weights.unsqueeze(-1) * span_token_embs).sum(dim=1)

        span_emb = torch.cat([start_vecs, end_vecs, head_vecs, width_vecs], dim=-1)

        if self.use_genre:
            genre_emb = self.get_genre_embedding(genre, device, span_emb.dtype)
            if genre_emb is None:
                genre_emb = torch.zeros(self.config["genre_emb_size"], device=device, dtype=span_emb.dtype)
            span_emb = torch.cat([span_emb, genre_emb.unsqueeze(0).expand(N_spans, -1)], dim=-1)

        span_emb = self.dropout(span_emb)

        #Passo 5: mention scoring e beam (global, não por segmento)
        mention_scores = self.get_mention_scores(span_emb)
        num_words = all_token_emb_cat.size(0)
        top_span_ratio = self.config.get("top_span_ratio", 0.4)
        k = max(1, min(3900, int(num_words * top_span_ratio), N_spans))
        c = min(self.config.get("max_top_antecedents", 50), k)

        top_scores, top_indices = torch.topk(mention_scores, k)

        # ordenar por posição no documento
        sort_order = torch.argsort(
            span_starts_tok[top_indices] * (num_words + 1) + span_ends_tok[top_indices]
        )
        top_indices    = top_indices[sort_order]
        top_scores     = top_scores[sort_order]

        span_emb_beam        = span_emb[top_indices]
        mention_scores_beam  = mention_scores[top_indices]
        span_starts_beam     = span_starts_tok[top_indices]
        span_ends_beam       = span_ends_tok[top_indices]
        span_segment_beam    = span_segment_ids[top_indices]

        #Passo 6: labels
        if gold_starts_all.numel() > 0:
            mention_labels_beam = self.get_candidate_labels(
                span_starts_beam, span_ends_beam, gold_starts_all.to(device), gold_ends_all.to(device)
            )
            candidate_cluster_ids_beam = self.get_candidate_cluster_ids(
                span_starts_beam, span_ends_beam,
                gold_starts_all.to(device), gold_ends_all.to(device),
                gold_cluster_ids_all.to(device)
            )
        else:
            mention_labels_beam       = torch.zeros(k, dtype=torch.long, device=device)
            candidate_cluster_ids_beam = torch.zeros(k, dtype=torch.long, device=device)

        # speakers
        beam_speaker_ids = None
        if self.use_speakers and speakers is not None:
            doc_speaker_flat = [spk for sent_spk in speakers for spk in sent_spk]
            doc_unique_spk   = list(dict.fromkeys(doc_speaker_flat))
            doc_spk_to_id    = {s: idx for idx, s in enumerate(doc_unique_spk)}
            raw_ids = []
            for s in span_starts_beam.tolist():
                if 0 <= s < len(doc_speaker_flat):
                    raw_ids.append(doc_spk_to_id.get(doc_speaker_flat[s], -1))
                else:
                    raw_ids.append(-1)
            beam_speaker_ids = torch.tensor(raw_ids, dtype=torch.long, device=device)

        #Passo 7: pair scoring global (UMA passagem, não por segmento)
        pair_scores = self.score_span_pairs(
            span_emb_beam, span_segment_beam, mention_scores_beam,
            speaker_ids=beam_speaker_ids
        )

        top_ant_scores, top_ant_idx = torch.topk(pair_scores, k=c, dim=1)
        logits = top_scores

        #Passo 8: loss
        loss = torch.zeros((), device=device, dtype=mention_scores.dtype)

        if mention_labels_beam.numel() > 0:
            n_pos = mention_labels_beam.sum().float().clamp(min=1.0)
            n_neg = (mention_labels_beam == 0).sum().float().clamp(min=1.0)
            pos_weight = (n_neg / n_pos).clamp(max=20.0)
            mention_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, mention_labels_beam.float(), pos_weight=pos_weight
            )
            loss = loss + mention_loss

        if candidate_cluster_ids_beam.numel() > 0:
            loss = loss + self.coref_marginal_loss_with_dummy(
                top_ant_scores, top_ant_idx, candidate_cluster_ids_beam
            )

        if return_debug:
            self.last_debug = {
                "top_scores":       top_scores.detach().cpu(),
                "span_starts_tok":  span_starts_beam.detach().cpu(),
                "span_ends_tok":    span_ends_beam.detach().cpu(),
                "pair_scores":      pair_scores.detach().cpu(),
                "span_emb":         span_emb_beam.detach(),
                "mention_scores":   mention_scores_beam.detach(),
                "span_segment_ids": span_segment_beam.detach(),
            }
        else:
            self.last_debug = None

        return logits, mention_labels_beam, loss
        
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
        speakers=None,
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
        
        # Filtrar spans em token space
        span_starts_tok = span_starts[keep_mask]
        span_ends_tok = span_ends[keep_mask]

        mention_labels = self.get_candidate_labels(
            span_starts_tok,
            span_ends_tok,
            gold_starts,
            gold_ends
        )

        candidate_cluster_ids = self.get_candidate_cluster_ids(
            span_starts_tok,
            span_ends_tok,
            gold_starts,
            gold_ends,
            gold_cluster_ids
        )

        span_segment_ids = span_segment_ids[keep_mask]
        #substitui os spans por wp space (consertando erro anterior BERT)
        span_starts = torch.tensor(span_start_wp, device=input_ids.device)
        span_ends = torch.tensor(span_end_wp, device=input_ids.device)
        
        assert span_starts.size(0) == mention_labels.size(0)
        
        #construindo speaker_ids (trabalhando com dados já padronizados)
        speaker_ids_tensor = None
        if self.use_speakers and speakers is not None:
            doc_speaker_flat = [spk for sent_spk in speakers for spk in sent_spk]
            doc_unique_spk = list(dict.fromkeys(doc_speaker_flat))
            doc_spk_to_id = {s: idx for idx, s in enumerate(doc_unique_spk)}

            seg_speaker_flat = [
                spk
                for sent_idx, sent_spk in enumerate(speakers)
                if seg_start <= sent_idx < seg_start + len(seg_sents)
                for spk in sent_spk
            ]

            raw_ids = []
            for s in span_starts_tok.tolist():
                if 0 <= s < len(seg_speaker_flat):
                    raw_ids.append(doc_spk_to_id.get(seg_speaker_flat[s], -1))
                else:
                    raw_ids.append(-1)

            speaker_ids_tensor = torch.tensor(raw_ids, dtype=torch.long)
    
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
            speaker_ids=speaker_ids_tensor.to(device) if speaker_ids_tensor is not None else None,
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
    def score_span_pairs(self, span_emb, span_segment_ids, mention_scores, speaker_ids=None):
        N, D = span_emb.size()
        device = span_emb.device
        dtype = span_emb.dtype

        #broadcasting: p interaçoes entre spans ao mesmo tempo, sem loop lento (expandir tensores)
        emb_i = span_emb.unsqueeze(1).expand(N, N, D)  #embedding do span i (menção)
        emb_j = span_emb.unsqueeze(0).expand(N, N, D)  #embedding do span j (antecedente)

        #distância de segmento -> embedding aprendivel (antecedente distante é menos provável)
        seg_i = span_segment_ids.unsqueeze(1).expand(N, N)
        seg_j = span_segment_ids.unsqueeze(0).expand(N, N)
        seg_dist = (seg_i - seg_j).clamp(0, self.max_training_sentences - 1)  
        seg_emb = self.segment_distance_embeddings(seg_dist) 

        pair_feats = [emb_i, emb_j, emb_i * emb_j, seg_emb]

        if self.use_speakers:
            if speaker_ids is not None:
                spk_i = speaker_ids.unsqueeze(1).expand(N, N)
                spk_j = speaker_ids.unsqueeze(0).expand(N, N)
                same = (spk_i == spk_j).long()
                unknown = ((spk_i < 0) | (spk_j < 0)).long()
                speaker_label = torch.where(
                    unknown == 1,
                    torch.full_like(same, 2),
                    1 - same
                )
            else:
                # sem info de speaker: tudo desconhecido
                speaker_label = torch.full((N, N), 2, dtype=torch.long, device=device)
            spk_emb = self.speaker_embeddings(speaker_label)
            pair_feats.append(spk_emb)

        pair_input = torch.cat(pair_feats, dim=-1)
    
        #um único forward pass (mais rapido -> loop anterior mt lento)
        pair_scores = self.pair_scorer(pair_input).squeeze(-1)

        pair_scores = pair_scores + mention_scores.unsqueeze(1) + mention_scores.unsqueeze(0) #adicionando mention scores

        #masking: só pares onde j < i (antecedentes válidos):
        mask = torch.ones(N, N, dtype=torch.bool, device=device).tril(diagonal=-1)
        pair_scores = pair_scores.masked_fill(~mask, -1e9)

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
        # Pegar o cluster_id do gold que bate com cada candidato.
        # Não usar "matched @ gold_cluster_ids" porque matmul com Long na CUDA quebra:
        # RuntimeError: "addmv_impl_cuda" not implemented for 'Long'
        has_match = eq.any(dim=1)
        cluster_ids = torch.zeros(N, dtype=torch.long, device=device)

        if has_match.any():
            match_idx = eq.float().argmax(dim=1)
            cluster_ids[has_match] = gold_cluster_ids.to(device).long()[match_idx[has_match]]

        return cluster_ids
    
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
    
    def coref_marginal_loss_with_dummy(self, top_scores, top_idx, candidate_cluster_ids):
        device = top_scores.device
        dtype = top_scores.dtype
        k = candidate_cluster_ids.size(0)

        if k == 0:
            return torch.tensor(0.0, device=device, dtype=dtype)

        # dummy score = 0 para todos os spans
        dummy = torch.zeros(k, 1, device=device, dtype=dtype)

        # distribuição completa: [dummy | top_c_antecedents] -> [k, c+1]
        all_scores = torch.cat([dummy, top_scores], dim=1)
        log_norm = torch.logsumexp(all_scores, dim=1)  

        # gold: antecedentes do top-c que têm o mesmo cluster_id
        cid_i = candidate_cluster_ids.unsqueeze(1)       
        cid_j = candidate_cluster_ids[top_idx]
        valid_antecedent = (top_scores > -1e8)
        same_cluster = (cid_i == cid_j) & (cid_i != 0) & valid_antecedent 

        gold_scores = top_scores.masked_fill(~same_cluster, -1e9)
        has_gold = same_cluster.any(dim=1)               

        log_gold_antecedent = torch.logsumexp(gold_scores, dim=1)
        log_gold = torch.where(has_gold, log_gold_antecedent, torch.zeros_like(log_norm))

        return (log_norm - log_gold).sum() 
