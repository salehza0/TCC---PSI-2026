import numpy as np
import pandas as pd
from pathlib import Path
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context
from src.task import DeepLSTM, gerar_dados_sinteticos, preparar_dados, get_weights, set_weights, train, evaluate

# Cache global — gerado uma vez por worker, reutilizado em todas as rodadas
_cache_clientes: dict = {}
DADOS_DIR = Path(r"C:\TCC\prova_conceito\dados")

class EstacaoMeteorologicaClient(NumPyClient):
    def __init__(self, partition_id, local_epochs, batch_size,
                 X_train, y_train, X_test, y_test, escala):
        self.partition_id = partition_id
        self.local_epochs = local_epochs
        self.batch_size   = batch_size
        self.model        = DeepLSTM()
        self.X_train      = X_train
        self.y_train      = y_train
        self.X_test       = X_test
        self.y_test       = y_test
        self.escala       = escala

    def fit(self, parameters, config):
        set_weights(self.model, parameters)
        epochs = int(config.get("local_epochs", self.local_epochs))
        lr     = float(config.get("lr", 0.001))
        loss   = train(self.model, self.X_train, self.y_train,
                       epochs=epochs, lr=lr, batch_size=self.batch_size)
        print(f"[Cliente {self.partition_id}] Treino concluído | Loss: {loss:.6f}")
        return get_weights(self.model), len(self.X_train), {"train_loss": float(loss)}

    def evaluate(self, parameters, config):
        set_weights(self.model, parameters)
        mse, rmse = evaluate(self.model, self.X_test, self.y_test)
        print(f"[Cliente {self.partition_id}] Avaliação | MSE: {mse:.6f} | RMSE: {rmse:.4f}°C")
        return float(mse), len(self.X_test), {"mse": float(mse), "rmse": float(rmse)}


# Cache — carrega o CSV apenas uma vez por worker
_cache_clientes: dict = {}

def client_fn(context: Context):
    partition_id = int(context.node_config["partition-id"])
    local_epochs = int(context.run_config.get("local-epochs", 3))
    batch_size   = int(context.run_config.get("batch-size", 64))

    if partition_id not in _cache_clientes:
        caminho = DADOS_DIR / f"estacao_{partition_id}.csv"
        df = pd.read_csv(caminho)
        X_train, y_train, X_test, y_test, escala = preparar_dados(df)
        _cache_clientes[partition_id] = (X_train, y_train, X_test, y_test, escala)
        print(f"[Cliente {partition_id}] CSV carregado: {caminho.name} | "
              f"treino={len(X_train)} | teste={len(X_test)}")

    X_train, y_train, X_test, y_test, escala = _cache_clientes[partition_id]

    return EstacaoMeteorologicaClient(
        partition_id=partition_id,
        local_epochs=local_epochs,
        batch_size=batch_size,
        X_train=X_train, y_train=y_train,
        X_test=X_test,   y_test=y_test,
        escala=escala,
    ).to_client()


app = ClientApp(client_fn=client_fn)