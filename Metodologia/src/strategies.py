"""
Estrategias configuraveis via pyproject.toml:
  - FedAvg    : Federated Averaging (classica)
  - FedMedian : Mediana robusta (tolerante a outliers)
  - FedYogi   : Adaptive aggregation com Yogi
  - FedAdam   : Adaptive aggregation com Adam
"""

from pathlib import Path
from flwr.common import parameters_to_ndarrays
from flwr.server.strategy import FedAvg, FedMedian, FedYogi, FedAdam


# -----------------------------------------------------------------------------
# Mixin de Salvamento (reutilizavel)
# -----------------------------------------------------------------------------

class _SalvamentoMixin:
    """Mixin que adiciona salvamento de pesos e metricas a qualquer estrategia."""

    def _init_salvamento(self, strategy_name: str, results_dir: Path, tracker=None):
        self.strategy_name = strategy_name
        self.results_dir = results_dir
        self.tracker = tracker
        self._pesos_rodada_atual = None

    def _salvar_pesos_fit(self, aggregated_parameters):
        """Captura pesos apos aggregate_fit."""
        if aggregated_parameters is not None:
            self._pesos_rodada_atual = parameters_to_ndarrays(aggregated_parameters)
            if self.tracker:
                self.tracker.capturar_pesos_agregados(aggregated_parameters)
            print(f"[{self.strategy_name}] Pesos capturados | "
                  f"{len(self._pesos_rodada_atual)} arrays")
        else:
            print(f"[{self.strategy_name}] AVISO: aggregated_parameters eh None!")

    def _imprimir_fit_clientes(self, server_round, results):
        """Imprime, uma linha por cliente, o loss de treino da rodada."""
        linhas = []
        for i, (_, fit_res) in enumerate(results):
            estacao = fit_res.metrics.get("estacao", f"estacao_{i}")
            loss = fit_res.metrics.get("train_loss", float("nan"))
            linhas.append(f"{estacao}: loss={loss:.6f}")
        linhas.sort()
        print(f"[Rodada {server_round}] Treino     -> " + " | ".join(linhas))

    def _processar_resultado(self, server_round, results, aggregated):
        """Imprime resultado por cliente e salva o agregado (delega ao tracker)."""
        if aggregated is None:
            return

        _, metrics_agregados = aggregated
        rmse = float(metrics_agregados.get("rmse", float("inf")))
        mse = float(metrics_agregados.get("mse", float("inf")))
        mae = float(metrics_agregados.get("mae", float("inf")))

        clientes_info = {}
        linhas = []
        for i, (client_proxy, evaluate_res) in enumerate(results):
            m = evaluate_res.metrics
            codigo_estacao = m.get("estacao", f"estacao_{i}")
            clientes_info[codigo_estacao] = {
                "mse": float(m.get("mse", 0)),
                "rmse": float(m.get("rmse", 0)),
                "mae": float(m.get("mae", 0)),
                "mse_norm": float(m.get("mse_norm", 0)),
                "rmse_norm": float(m.get("rmse_norm", 0)),
                "mae_norm": float(m.get("mae_norm", 0)),
                "num_amostras": evaluate_res.num_examples,
            }
            linhas.append(f"{codigo_estacao}: RMSE={m.get('rmse', 0):.4f}C MAE={m.get('mae', 0):.4f}C")

        linhas.sort()
        print(f"[Rodada {server_round}] Avaliacao  -> " + " | ".join(linhas))

        if self.tracker:
            self.tracker.registrar_rodada(server_round, rmse, mse, clientes_info, mae)
        else:
            print(f"[{self.strategy_name}] Rodada {server_round}: "
                  f"RMSE={rmse:.4f}C | MSE={mse:.6f} | MAE={mae:.4f}C")


# -----------------------------------------------------------------------------
# Mixin generico de aggregate_fit / aggregate_evaluate
# -----------------------------------------------------------------------------
#
# FedAvg, FedMedian, FedYogi e FedAdam tem a MESMA assinatura de
# aggregate_fit/aggregate_evaluate (todas herdam de Strategy). Por isso o
# comportamento de "capturar pesos" e "processar resultado" pode ser implementado de forma generica aqui.
# -----------------------------------------------------------------------------

class _AggregateComSalvamento(_SalvamentoMixin):
    """Mixin com aggregate_fit/aggregate_evaluate genericos, delegando a
    agregacao real (a matematica) para a classe base do Flower via super()."""

    def aggregate_fit(self, server_round, results, failures):
        self._imprimir_fit_clientes(server_round, results)
        aggregated = super().aggregate_fit(server_round, results, failures)
        if aggregated is not None:
            self._salvar_pesos_fit(aggregated[0])
        return aggregated

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated = super().aggregate_evaluate(server_round, results, failures)
        self._processar_resultado(server_round, results, aggregated)
        return aggregated


# -----------------------------------------------------------------------------
# Estrategias Concretas - cada uma so troca a classe base do Flower
# -----------------------------------------------------------------------------

class FedAvgComSalvamento(_AggregateComSalvamento, FedAvg):
    """FedAvg (flwr.server.strategy.FedAvg)"""

    def __init__(self, strategy_name: str, results_dir: Path, tracker=None, **kwargs):
        FedAvg.__init__(self, **kwargs)
        self._init_salvamento(strategy_name, results_dir, tracker)


class FedMedianComSalvamento(_AggregateComSalvamento, FedMedian):
    """FedMedian (flwr.server.strategy.FedMedian)"""

    def __init__(self, strategy_name: str, results_dir: Path, tracker=None, **kwargs):
        FedMedian.__init__(self, **kwargs)
        self._init_salvamento(strategy_name, results_dir, tracker)


class FedYogiComSalvamento(_AggregateComSalvamento, FedYogi):
    """
    FedYogi (flwr.server.strategy.FedYogi)
    Parametros extras aceitos (repassados ao FedYogi nativo via **kwargs):
      eta, eta_l, beta_1, beta_2, tau  (ver docs do Flower)
    """

    def __init__(self, strategy_name: str, results_dir: Path, tracker=None, **kwargs):
        FedYogi.__init__(self, **kwargs)
        self._init_salvamento(strategy_name, results_dir, tracker)


class FedAdamComSalvamento(_AggregateComSalvamento, FedAdam):
    """
    FedAdam (flwr.server.strategy.FedAdam)
    Parametros extras aceitos (repassados ao FedAdam nativo via **kwargs):
      eta, eta_l, beta_1, beta_2, tau
    """

    def __init__(self, strategy_name: str, results_dir: Path, tracker=None, **kwargs):
        FedAdam.__init__(self, **kwargs)
        self._init_salvamento(strategy_name, results_dir, tracker)


# -----------------------------------------------------------------------------
# Factory Function
# -----------------------------------------------------------------------------

STRATEGIES_DISPONIBLES = {
    "fedavg": FedAvgComSalvamento,
    "fedmedian": FedMedianComSalvamento,
    "fedyogi": FedYogiComSalvamento,
    "fedadam": FedAdamComSalvamento,
}


def create_strategy(strategy_name: str, results_dir: Path, tracker=None, **kwargs):
    """
    Funcao para criar estrategias.

    """
    strategy_name = strategy_name.lower().strip()

    if strategy_name not in STRATEGIES_DISPONIBLES:
        raise ValueError(
            f"Estrategia '{strategy_name}' nao reconhecida. "
            f"Disponiveis: {list(STRATEGIES_DISPONIBLES.keys())}"
        )

    Strategy = STRATEGIES_DISPONIBLES[strategy_name]
    return Strategy(strategy_name=strategy_name, results_dir=results_dir, tracker=tracker, **kwargs)