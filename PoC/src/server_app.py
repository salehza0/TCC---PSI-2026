"""
server_app.py — Define o ServerApp para aprendizado federado.

Suporta duas estratégias de agregação configuráveis via pyproject.toml:
  - FedAvg  : Federated Averaging (clássica)
"""
import os
import json
import numpy as np
from pathlib import Path

import torch
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays, Metrics
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg

from src.task import DeepLSTM, get_weights, set_weights


# ---------------------------------------------------------------------------
# Mixin reutilizável com lógica de salvamento
# ---------------------------------------------------------------------------
class _SalvamentoMixin:
    """Mixin que adiciona salvamento de pesos e métricas a qualquer estratégia."""

    def _init_salvamento(self, strategy_name: str, results_dir: Path):
        self.strategy_name = strategy_name
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.historico: list[dict] = []
        self.melhor_rmse = float("inf")
        self.melhor_rodada = -1
        self._pesos_rodada_atual = None  # ← armazena pesos após aggregate_fit
        self.arquivo_pesos = results_dir / f"melhores_pesos_{strategy_name}.pt"
        self.arquivo_metricas = results_dir / f"metricas_{strategy_name}.json"

    def _salvar_pesos_fit(self, aggregated_parameters):
        if aggregated_parameters is not None:
            self._pesos_rodada_atual = parameters_to_ndarrays(aggregated_parameters)
            print(f"[Servidor] Pesos capturados | {len(self._pesos_rodada_atual)} arrays")
        else:
            print(f"[Servidor] AVISO: aggregated_parameters é None!")

    def _processar_resultado(self, server_round, results, aggregated):
        if aggregated is None:
            return
        _, metrics_agregados = aggregated
        rmse = metrics_agregados.get("rmse", float("inf"))
        mse  = metrics_agregados.get("mse",  float("inf"))

        clientes_info = {}
        for i, (client_proxy, evaluate_res) in enumerate(results):
            clientes_info[f"estacao_{i}"] = {
                "mse":          float(evaluate_res.metrics.get("mse", 0)),
                "rmse":         float(evaluate_res.metrics.get("rmse", 0)),
                "num_amostras": evaluate_res.num_examples,
            }

        entrada = {
            "rodada":         server_round,
            "rmse_global":    float(rmse),
            "mse_global":     float(mse),
            "melhor_ate_agora": False,
            "clientes":       clientes_info,
        }

        # Salva os melhores pesos usando os pesos capturados no aggregate_fit
        if rmse < self.melhor_rmse and self._pesos_rodada_atual is not None:
            self.melhor_rmse   = rmse
            self.melhor_rodada = server_round
            entrada["melhor_ate_agora"] = True

            modelo = DeepLSTM()
            set_weights(modelo, self._pesos_rodada_atual)
            torch.save({
                "rodada":           server_round,
                "rmse":             float(rmse),
                "strategy":         self.strategy_name,
                "model_state_dict": modelo.state_dict(),
            }, self.arquivo_pesos)
            print(f"[Servidor] Melhores pesos salvos! Rodada {server_round} | RMSE: {rmse:.4f}°C")

        self.historico.append(entrada)
        with open(self.arquivo_metricas, "w", encoding="utf-8") as f:
            json.dump({
                "strategy":       self.strategy_name,
                "total_rodadas":  server_round,
                "melhor_rodada":  self.melhor_rodada,
                "melhor_rmse":    float(self.melhor_rmse),
                "historico":      self.historico,
            }, f, indent=2, ensure_ascii=False)
            
class FedAvgComSalvamento(_SalvamentoMixin, FedAvg):
    def __init__(self, strategy_name, results_dir, **kwargs):
        FedAvg.__init__(self, **kwargs)
        self._init_salvamento(strategy_name, results_dir)

    def aggregate_fit(self, server_round, results, failures):
        aggregated = super().aggregate_fit(server_round, results, failures)
        if aggregated is not None:
            self._salvar_pesos_fit(aggregated[0])  # ← captura pesos aqui
        return aggregated

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated = super().aggregate_evaluate(server_round, results, failures)
        self._processar_resultado(server_round, results, aggregated)
        return aggregated

# ---------------------------------------------------------------------------
# Funcao de agregacao de metricas
# ---------------------------------------------------------------------------

def weighted_average_metrics(metrics: list[tuple[int, Metrics]]) -> Metrics:
    total_amostras = sum(n for n, _ in metrics)
    mse_agregado = sum(n * m["mse"] for n, m in metrics) / total_amostras
    rmse_agregado = float(np.sqrt(mse_agregado))
    print(f"\n[Servidor] Metricas agregadas | MSE: {mse_agregado:.6f} | RMSE: {rmse_agregado:.4f}°C\n")
    return {"mse": mse_agregado, "rmse": rmse_agregado}


# ---------------------------------------------------------------------------
# Fabrica do ServerApp
# ---------------------------------------------------------------------------

def server_fn(context: Context) -> ServerAppComponents:
    # Usa variável de ambiente ou fallback para o caminho fixo
    # project_dir = Path(os.environ.get(
    #     "FLWR_PROJECT_DIR",
    #     r"C:\TCC\prova_conceito"
    # ))
    # results_dir = project_dir / "results"
    # results_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(r"C:\TCC\prova_conceito\results")
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Servidor] Salvando em: {results_dir.resolve()}")
    num_rounds = int(context.run_config.get("num-server-rounds", 30))
    # estrategia_nome = str(context.run_config.get("strategy", "fedavg")).lower()
    fraction_fit = float(context.run_config.get("fraction-fit", 1.0))
    fraction_evaluate = float(context.run_config.get("fraction-evaluate", 1.0))
    # mu = float(context.run_config.get("mu", 0.01))
    print(f"[Servidor] Salvando resultados em: {results_dir.resolve()}")
    modelo_global = DeepLSTM()
    pesos_iniciais = ndarrays_to_parameters(get_weights(modelo_global))

    kwargs_comuns = dict(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        initial_parameters=pesos_iniciais,
        evaluate_metrics_aggregation_fn=weighted_average_metrics,
    )

    estrategia = FedAvgComSalvamento(
            strategy_name="fedavg",
            results_dir=results_dir,
            **kwargs_comuns,
        )
    print(f"[Servidor] Estrategia: FedAvg | rodadas={num_rounds}")

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=estrategia, config=config)


app = ServerApp(server_fn=server_fn)
