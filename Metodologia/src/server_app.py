import numpy as np
from pathlib import Path
from flwr.common import (
    Context, ndarrays_to_parameters, parameters_to_ndarrays, Metrics
)
from flwr.server import ServerApp, ServerConfig, ServerAppComponents

from src.task import criar_modelo, get_weights, set_weights, N_FEATURES
from src.strategies import create_strategy, STRATEGIES_DISPONIBLES

# Importa ExperimentTracker se disponivel
try:
    from src.utils.experiment_tracker import ExperimentTracker
    HAS_TRACKER = True
except ImportError:
    HAS_TRACKER = False
    print("[AVISO] ExperimentTracker nao encontrado. Usando modo legado.")


# -----------------------------------------------------------------------------
# Funcao de agregacao de metricas (reutilizavel)
# -----------------------------------------------------------------------------

def weighted_average_metrics(metrics: list[tuple[int, Metrics]]) -> Metrics:
    """
    Agrega metricas de clientes usando media ponderada pelo numero de amostras.
    
    Args:
        metrics: lista de tuplas (num_amostras, {metrica: valor})
    
    Returns:
        Dicionario com metricas agregadas
    """
    vazio = {"mse": float("inf"), "rmse": float("inf"), "mae": float("inf"),
             "mse_norm": float("inf"), "rmse_norm": float("inf"), "mae_norm": float("inf")}

    if not metrics:
        return vazio

    total_amostras = sum(n for n, _ in metrics)
    if total_amostras == 0:
        return vazio

    mse_agregado = sum(n * m.get("mse", 0) for n, m in metrics) / total_amostras
    rmse_agregado = float(np.sqrt(mse_agregado))
    mae_agregado = sum(n * m.get("mae", 0) for n, m in metrics) / total_amostras

    mse_norm_agregado = sum(n * m.get("mse_norm", 0) for n, m in metrics) / total_amostras
    rmse_norm_agregado = float(np.sqrt(mse_norm_agregado))
    mae_norm_agregado = sum(n * m.get("mae_norm", 0) for n, m in metrics) / total_amostras

    print(f"\n[Servidor] Metricas agregadas | "
          f"Normalizado: MSE={mse_norm_agregado:.6f} RMSE={rmse_norm_agregado:.6f} MAE={mae_norm_agregado:.6f} | "
          f"C: MSE={mse_agregado:.6f} RMSE={rmse_agregado:.4f}C MAE={mae_agregado:.4f}C\n")

    return {
        "mse": mse_agregado,
        "rmse": rmse_agregado,
        "mae": mae_agregado,
        "mse_norm": mse_norm_agregado,
        "rmse_norm": rmse_norm_agregado,
        "mae_norm": mae_norm_agregado,
    }


# -----------------------------------------------------------------------------
# Funcao Principal do Servidor
# -----------------------------------------------------------------------------

def server_fn(context: Context) -> ServerAppComponents:
    """
    Factory do ServerApp com suporte a multiplas estrategias.
    Configuracao via pyproject.toml

    """
    
    # -------------------------------------------------------------------------
    # 1. Setup de diretorios
    # -------------------------------------------------------------------------
    
    results_dir = Path(r"C:\Users\nasse\OneDrive\Documentos\TCC\Metodologia\results\Federado")
    
    # -------------------------------------------------------------------------
    # 2. Le configuracao
    # -------------------------------------------------------------------------
    
    num_rounds = int(context.run_config.get("num-server-rounds", 30))
    strategy_nome = str(context.run_config.get("strategy", "fedavg")).lower().strip()
    model_nome = str(context.run_config.get("model", "lstm")).lower().strip()
    fraction_fit = float(context.run_config.get("fraction-fit", 1.0))
    fraction_evaluate = float(context.run_config.get("fraction-evaluate", 1.0))

    # Valida estrategia
    if strategy_nome not in STRATEGIES_DISPONIBLES:
        raise ValueError(
            f"Estrategia '{strategy_nome}' nao reconhecida. "
            f"Disponiveis: {list(STRATEGIES_DISPONIBLES.keys())}"
        )

    print(f"[Servidor] Estrategia: {strategy_nome.upper()}")
    print(f"[Servidor] Modelo: {model_nome.upper()}")
    print(f"[Servidor] Rodadas: {num_rounds}")
    print(f"[Servidor] Fraction Fit: {fraction_fit} | Fraction Evaluate: {fraction_evaluate}\n")

    # -------------------------------------------------------------------------
    # 3. Inicializa modelo global e pesos iniciais
    # -------------------------------------------------------------------------

    modelo_global = criar_modelo(model_nome)
    pesos_iniciais = ndarrays_to_parameters(get_weights(modelo_global))
    
    # -------------------------------------------------------------------------
    # 4. Inicializa ExperimentTracker (se disponivel)
    # -------------------------------------------------------------------------
    
    tracker = None
    if HAS_TRACKER:
        tracker = ExperimentTracker(results_dir, strategy_nome, model_nome)
        tracker.salvar_config({
            "strategy": strategy_nome,
            "model": model_nome,
            "num_rounds": num_rounds,
            "local_epochs": context.run_config.get("local-epochs"),
            "batch_size": context.run_config.get("batch-size"),
            "learning_rate": context.run_config.get("lr"),
            "fraction_fit": fraction_fit,
            "fraction_evaluate": fraction_evaluate,
            "n_features": N_FEATURES,
        })
        print(f"[Servidor] ExperimentTracker ativado: {tracker.experiment_dir.name}\n")
    
    # -------------------------------------------------------------------------
    # 5. Argumentos comuns para qualquer estrategia
    # -------------------------------------------------------------------------
    
    kwargs_comuns = dict(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        initial_parameters=pesos_iniciais,
        evaluate_metrics_aggregation_fn=weighted_average_metrics,
    )

    # Hiperparametros especificos do FedOpt (FedAdam/FedYogi), configuraveis
    # via pyproject.toml. Defaults conforme a documentacao do Flower.
    if strategy_nome in ("fedadam", "fedyogi"):
        kwargs_comuns.update(
            eta=float(context.run_config.get("eta", 0.1)),
            eta_l=float(context.run_config.get("eta-l", 0.1)),
            beta_1=float(context.run_config.get("beta-1", 0.9)),
            beta_2=float(context.run_config.get("beta-2", 0.99)),
            tau=float(context.run_config.get("tau", 1e-9)),
        )

    # -------------------------------------------------------------------------
    # 6. Cria estrategia via factory
    # -------------------------------------------------------------------------
    
    estrategia = create_strategy(
        strategy_name=strategy_nome,
        results_dir=results_dir,
        tracker=tracker,
        **kwargs_comuns
    )
    
    print(f"[Servidor] Estrategia criada: {strategy_nome.upper()}\n")
    
    # -------------------------------------------------------------------------
    # 7. Configuracao final e retorno
    # -------------------------------------------------------------------------
    
    config = ServerConfig(num_rounds=num_rounds)
    
    return ServerAppComponents(strategy=estrategia, config=config)


# Cria o ServerApp
app = ServerApp(server_fn=server_fn)
