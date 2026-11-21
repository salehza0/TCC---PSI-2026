import numpy as np
import pandas as pd
from pathlib import Path
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context
from src.task import criar_modelo, preparar_dados, get_weights, set_weights, train, evaluate

# Cache global - gerado uma vez por worker, reutilizado em todas as rodadas
_cache_clientes: dict = {}
DADOS_DIR = Path(r"C:\Users\nasse\OneDrive\Documentos\TCC\Metodologia\dados")

# Mapeamento partition_id -> arquivo de estacao (ordem fixa entre as rodadas)
ARQUIVOS_ESTACOES = [
    "A701 - SAO PAULO - MIRANTE.csv",
    "A706 - CAMPOS DO JORDAO.csv",
    "A744 - BRAGANCA PAULISTA.csv",
    "A755 - BARUERI.csv",
    "A771 - SAO PAULO - INTERLAGOS.csv",
]

class EstacaoMeteorologicaClient(NumPyClient):
    def __init__(self, partition_id, codigo_estacao, model_nome, local_epochs, batch_size,
                 X_train, y_train, X_test, y_test, escala):
        self.partition_id  = partition_id
        self.codigo_estacao = codigo_estacao
        self.local_epochs  = local_epochs
        self.batch_size    = batch_size
        self.model         = criar_modelo(model_nome)
        self.X_train       = X_train
        self.y_train       = y_train
        self.X_test        = X_test
        self.y_test        = y_test
        self.escala        = escala

    def fit(self, parameters, config):
        set_weights(self.model, parameters)
        epochs = int(config.get("local_epochs", self.local_epochs))
        lr     = float(config.get("lr", 0.001))
        loss   = train(self.model, self.X_train, self.y_train,
                       epochs=epochs, lr=lr, batch_size=self.batch_size)
        return get_weights(self.model), len(self.X_train), {
            "estacao": self.codigo_estacao,
            "train_loss": float(loss),
        }

    def evaluate(self, parameters, config):
        set_weights(self.model, parameters)
        metricas = evaluate(self.model, self.X_test, self.y_test, escala=self.escala)
        return float(metricas["mse_norm"]), len(self.X_test), {
            "estacao": self.codigo_estacao,
            "mse_norm": float(metricas["mse_norm"]),
            "rmse_norm": float(metricas["rmse_norm"]),
            "mae_norm": float(metricas["mae_norm"]),
            "mse": float(metricas["mse_celsius"]),
            "rmse": float(metricas["rmse_celsius"]),
            "mae": float(metricas["mae_celsius"]),
            "r2": float(metricas["r2"]),
        }


# Cache - carrega o CSV apenas uma vez por worker
_cache_clientes: dict = {}

def client_fn(context: Context):
    partition_id = int(context.node_config["partition-id"])
    local_epochs = int(context.run_config.get("local-epochs", 1))
    batch_size   = int(context.run_config.get("batch-size", 64))
    model_nome   = str(context.run_config.get("model", "lstm")).lower().strip()

    nome_arquivo = ARQUIVOS_ESTACOES[partition_id]
    codigo_estacao = nome_arquivo.split(" - ")[0] 

    if partition_id not in _cache_clientes:
        caminho = DADOS_DIR / nome_arquivo
        df = pd.read_csv(caminho, index_col="timestamp", parse_dates=True)
        if partition_id==0:
            df = df.loc[df.index >= "2024-01-01"] ## Teste
        elif partition_id==1:
            df = df.loc[df.index >= "2024-06-01"] ## Teste
        elif partition_id==2:
            df = df.loc[df.index >= "2025-01-01"] ## Teste
        elif partition_id==3:
            df = df.loc[df.index >= "2025-06-01"] ## Teste
        elif partition_id==4:
            df = df.loc[df.index >= "2025-10-01"] ## Teste
        X_train, y_train, X_test, y_test, escala = preparar_dados(df, verbose=False)
        _cache_clientes[partition_id] = (X_train, y_train, X_test, y_test, escala)
        print(f"[Cliente {codigo_estacao}] CSV carregado: {caminho.name} | "
              f"treino={len(X_train)} | teste={len(X_test)}")

    X_train, y_train, X_test, y_test, escala = _cache_clientes[partition_id]

    return EstacaoMeteorologicaClient(
        partition_id=partition_id,
        codigo_estacao=codigo_estacao,
        model_nome=model_nome,
        local_epochs=local_epochs,
        batch_size=batch_size,
        X_train=X_train, y_train=y_train,
        X_test=X_test,   y_test=y_test,
        escala=escala,
    ).to_client()


app = ClientApp(client_fn=client_fn)