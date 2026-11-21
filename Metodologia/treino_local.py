import argparse
import copy
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from src.task import DeepLSTM, TCN, preparar_dados, train, evaluate

DADOS_DIR = Path(__file__).resolve().parent / "dados"
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "Local"

MODELOS = {"lstm": DeepLSTM, "tcn": TCN}


def carregar_estacoes(dados_dir: Path) -> dict[str, pd.DataFrame]:
    """Carrega um DataFrame por estacao a partir dos CSVs em dados_dir (ignora dados_gerais.csv)."""
    estacoes = {}
    for arquivo in sorted(dados_dir.glob("*.csv")):
        if arquivo.stem == "dados_gerais":
            continue
        codigo = arquivo.stem.split(" - ")[0].strip()
        estacoes[codigo] = pd.read_csv(arquivo, parse_dates=["timestamp"], index_col="timestamp")
        if codigo=="A701":
            estacoes[codigo] = estacoes[codigo].loc[estacoes[codigo].index >= "2024-01-01"] ## Teste
        if codigo=="A706":
            estacoes[codigo] = estacoes[codigo].loc[estacoes[codigo].index >= "2024-06-01"] ## Teste
        if codigo=="A744":
            estacoes[codigo] = estacoes[codigo].loc[estacoes[codigo].index >= "2025-01-01"] ## Teste
        if codigo=="A755":
            estacoes[codigo] = estacoes[codigo].loc[estacoes[codigo].index >= "2025-06-01"] ## Teste
        if codigo=="A771":
            estacoes[codigo] = estacoes[codigo].loc[estacoes[codigo].index >= "2025-10-01"] ## Teste
    return estacoes


def treinar_com_checkpoint(model_class, df: pd.DataFrame, epochs: int, lr: float,
                            batch_size: int, split: float):
    """Treina epoca a epoca e retorna os pesos da epoca com menor RMSE (C) no teste."""
    X_train, y_train, X_test, y_test, escala = preparar_dados(df, split=split)
    modelo = model_class(n_features=X_train.shape[-1])

    melhor_rmse = float("inf")
    melhores_pesos = None
    melhor_epoca = -1
    historico = []

    for epoca in range(1, epochs + 1):
        loss_treino = train(modelo, X_train, y_train, epochs=1, lr=lr, batch_size=batch_size)
        metricas = evaluate(modelo, X_test, y_test, batch_size=batch_size, escala=escala)
        historico.append({"epoca": epoca, "loss_treino": loss_treino, **metricas})

        melhorou = metricas["rmse_celsius"] < melhor_rmse
        if melhorou:
            melhor_rmse = metricas["rmse_celsius"]
            melhor_epoca = epoca
            melhores_pesos = copy.deepcopy(modelo.state_dict())

        marcador = " *melhor*" if melhorou else ""
        print(f"  Epoca {epoca:>3}/{epochs} | loss_treino={loss_treino:.6f} | "
              f"RMSE={metricas['rmse_celsius']:.4f}C{marcador}")

    modelo.load_state_dict(melhores_pesos)
    metricas_finais = evaluate(modelo, X_test, y_test, batch_size=batch_size, escala=escala)

    return melhores_pesos, melhor_epoca, historico, metricas_finais


def main():
    parser = argparse.ArgumentParser(description="Treino local: um modelo por estacao")
    parser.add_argument("--modelos", nargs="+", choices=MODELOS.keys(), default=list(MODELOS.keys()),
                         help="Modelos a treinar nesta rodada (default: todos)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--split", type=float, default=0.8)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    config = {
        "modelos": args.modelos,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "split": args.split,
        "timestamp": timestamp,
    }

    estacoes = carregar_estacoes(DADOS_DIR)
    print(f"[Local] {len(estacoes)} estacao(oes) | modelos: {', '.join(args.modelos)}")

    resumo_geral = {"config": config, "estacoes": {}}

    for codigo, df in estacoes.items():
        resumo_geral["estacoes"][codigo] = {}

        for nome_modelo in args.modelos:
            print(f"\n=== Estacao {codigo} | {nome_modelo.upper()} ===")
            pesos, melhor_epoca, historico, metricas = treinar_com_checkpoint(
                MODELOS[nome_modelo], df, args.epochs, args.lr, args.batch_size, args.split
            )

            # Uma pasta por estacao; dentro, uma subpasta por execucao (tag do modelo + timestamp)
            run_dir = RESULTS_DIR / codigo / f"{nome_modelo}_{timestamp}"
            run_dir.mkdir(parents=True, exist_ok=True)

            torch.save({
                "codigo": codigo,
                "modelo": nome_modelo,
                "epoca": melhor_epoca,
                "model_state_dict": pesos,
                "metricas": metricas,
            }, run_dir / "melhor.pt")

            with open(run_dir / "metricas.json", "w", encoding="utf-8") as f:
                json.dump({
                    "codigo": codigo,
                    "modelo": nome_modelo,
                    "melhor_epoca": melhor_epoca,
                    "metricas": metricas,
                    "historico": historico,
                }, f, indent=2, ensure_ascii=False)

            print(f"[Local] {codigo} [{nome_modelo}]: melhor RMSE={metricas['rmse_celsius']:.4f}C "
                  f"(epoca {melhor_epoca}/{args.epochs}) | salvo em {run_dir.relative_to(RESULTS_DIR)}")

            resumo_geral["estacoes"][codigo][nome_modelo] = {
                "melhor_epoca": melhor_epoca,
                "metricas": metricas,
            }

    with open(RESULTS_DIR / f"resumo_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(resumo_geral, f, indent=2, ensure_ascii=False)

    print(f"\n[Local] Resultados salvos em: {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
