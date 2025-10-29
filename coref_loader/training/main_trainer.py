#imprime instâncias do dataset
    
#dificuldades: 
#tentei usar o dataset original do OntoNotes, com o codigo ontonotes_to_jsonlines.py, mas tive problemas com formatos: ele vem em formato .onf e o ontonotes-db-tool oficial é em Python 2, então não consegui importar.
        #mudei de plano e passei a testar com arquivos .jsonlines criados manualmente.
#no setup inicial: tive muitos erros de ambiente (versões de python, tensorflow, etc) -> resolvi criando um ambiente limpo (python 3.7) e instalando manualmente o que precisava.
#imports e estrutura: o python não encontrava o pacote coref_loader -> resolvi rodando o trainer como módulo (python -m coref_loader.training.main_trainer).
#roda 100% em Python 3; consigo ler e imprimir instâncias do dataset no formato .jsonlines; o código está sem dependências antigas (tensorflow, pytorch, etc.)

import argparse #biblio p rodar o código direto no terminal
from pathlib import Path
from coref_loader.data import CorefDataset

#contando quantos tokens existem no total em todas as sentences do doc:
def count_tokens(sentences):
    return sum(len(s) for s in sentences)

def main():
    ap = argparse.ArgumentParser() #criando o leitor de argumentos. add os args que o programa vai aceitar:
    ap.add_argument("--data_dir", required=True, help="Pasta que contém *.english.jsonlines") #precisa do datadir (qual pasta p abrir)
    ap.add_argument("--split", default="dev", choices=["train", "dev", "test"], #define o split (qual arq -padrão é dev)
                    help="Qual arquivo abrir (train/dev/test)")
    ap.add_argument("--limit", type=int, default=20, help="Quantos docs imprimir (0 = todos)") #quantos docs quero ver
    ap.add_argument("--preview", type=int, default=12, help="Qtde de tokens para prévia da 1ª sentença") #quantos tokens quero mostrar de exemplo
    args = ap.parse_args() #le o que foi digitado no terminal e guarda em args

    ds = CorefDataset(args.data_dir, args.split) #criao conjunto de dados com os args passados
    print(f"[OK] carregado: {len(ds)} documentos ({args.split})\n") #imprime n de docs que foram carregados

    n = len(ds) if args.limit == 0 else min(args.limit, len(ds)) #pra saber quantos imprimir: 0 = todos / menor limite entre o pedido e o n total
    for i in range(n): #infos basicas do doc:
        doc = ds[i]
        doc_key = doc["doc_key"]
        nsents = len(doc["sentences"])
        ntoks = count_tokens(doc["sentences"])
        nclus = len(doc.get("clusters", []))
        
        preview = " ".join(doc["sentences"][0][:args.preview]) if nsents else "" #junta os primeiros args.preview tokens da primeira sentence p mostrar
        print(f"[{i:05d}] {doc_key} | sentenças: {nsents} | tokens_total: {ntoks} | clusters: {nclus}") #formato para imprimir
        if preview:
            print(f"       1ª sentença: {preview}")
    print("\n[done]")

if __name__ == "__main__":
    main()
