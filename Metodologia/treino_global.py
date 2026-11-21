import argparse
import copy
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from src.task import DeepLSTM, TCN, preparar_dados, train, evaluate

DADOS_DIR = Path(__file__).resolve().parent / "dados"
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "Global"

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


def preparar_dados_globais(estacoes: dict[str, pd.DataFrame], split: float):
    """Prepara cada estacao separadamente e concatena os tensores num unico conjunto treino/teste."""
    codigos, X_trains, y_trains, X_tests, y_tests, escalas = [], [], [], [], [], []

    for codigo, df in estacoes.items():
        X_train, y_train, X_test, y_test, escala = preparar_dados(df, split=split)
        codigos.append(codigo)
        X_trains.append(X_train)
        y_trains.append(y_train)
        X_tests.append(X_test)
        y_tests.append(y_test)
        escalas.append(escala)

    X_train = torch.cat(X_trains, dim=0)
    y_train = torch.cat(y_trains, dim=0)
    X_test = torch.cat(X_tests, dim=0)
    y_test = torch.cat(y_tests, dim=0)

    return X_train, y_train, X_test, y_test, codigos, X_tests, y_tests, escalas


def treinar_com_checkpoint(model_class, X_train, y_train, X_test, y_test,
                            epochs: int, lr: float, batch_size: int):
    """Treina epoca a epoca e retorna o modelo com os pesos da epoca de menor RMSE normalizado no teste."""
    modelo = model_class(n_features=X_train.shape[-1])

    melhor_rmse = float("inf")
    melhores_pesos = None
    melhor_epoca = -1
    historico = []

    for epoca in range(1, epochs + 1):
        loss_treino = train(modelo, X_train, y_train, epochs=1, lr=lr, batch_size=batch_size)
        metricas = evaluate(modelo, X_test, y_test, batch_size=batch_size)
        historico.append({"epoca": epoca, "loss_treino": loss_treino, **metricas})

        melhorou = metricas["rmse_norm"] < melhor_rmse
        if melhorou:
            melhor_rmse = metricas["rmse_norm"]
            melhor_epoca = epoca
            melhores_pesos = copy.deepcopy(modelo.state_dict())

        marcador = " *melhor*" if melhorou else ""
        print(f"  Epoca {epoca:>3}/{epochs} | loss_treino={loss_treino:.6f} | "
              f"RMSE_norm={metricas['rmse_norm']:.6f}{marcador}")

    modelo.load_state_dict(melhores_pesos)
    return modelo, melhores_pesos, melhor_epoca, historico


def main():
    parser = argparse.ArgumentParser(description="Treino global")
    parser.add_argument("--modelos", nargs="+", choices=MODELOS.keys(), default=list(MODELOS.keys()),
                         help="Modelos a treinar nesta rodada (default: todos)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--split", type=float, default=0.8)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    estacoes = carregar_estacoes(DADOS_DIR)
    print(f"[Global] {len(estacoes)} estacao(oes) | modelos: {', '.join(args.modelos)}")

    X_train, y_train, X_test, y_test, codigos, X_tests, y_tests, escalas = \
        preparar_dados_globais(estacoes, args.split)
    print(f"[Global] Dados combinados | treino={len(X_train)} | teste={len(X_test)}")

    resumo_geral = {
        "config": {
            "modelos": args.modelos,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "split": args.split,
            "timestamp": timestamp,
            "estacoes": codigos,
        },
        "modelos": {},
    }

    for nome_modelo in args.modelos:
        print(f"\n=== Modelo {nome_modelo.upper()} ===")
        modelo, pesos, melhor_epoca, historico = treinar_com_checkpoint(
            MODELOS[nome_modelo], X_train, y_train, X_test, y_test,
            args.epochs, args.lr, args.batch_size,
        )

        metricas_globais = evaluate(modelo, X_test, y_test, batch_size=args.batch_size)
        metricas_por_estacao = {
            codigo: evaluate(modelo, X_t, y_t, batch_size=args.batch_size, escala=esc)
            for codigo, X_t, y_t, esc in zip(codigos, X_tests, y_tests, escalas)
        }

        # Uma subpasta por execucao (tag do modelo + timestamp)
        run_dir = RESULTS_DIR / f"{nome_modelo}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        torch.save({
            "modelo": nome_modelo,
            "epoca": melhor_epoca,
            "model_state_dict": pesos,
            "metricas_globais": metricas_globais,
            "metricas_por_estacao": metricas_por_estacao,
        }, run_dir / "melhor.pt")

        with open(run_dir / "metricas.json", "w", encoding="utf-8") as f:
            json.dump({
                "modelo": nome_modelo,
                "estacoes": codigos,
                "melhor_epoca": melhor_epoca,
                "metricas_globais": metricas_globais,
                "metricas_por_estacao": metricas_por_estacao,
                "historico": historico,
            }, f, indent=2, ensure_ascii=False)

        print(f"[Global] {nome_modelo}: melhor RMSE normalizado={metricas_globais['rmse_norm']:.6f} "
              f"(epoca {melhor_epoca}/{args.epochs}) | salvo em {run_dir.relative_to(RESULTS_DIR)}")
        for codigo, m in metricas_por_estacao.items():
            print(f"  {codigo}: RMSE={m['rmse_celsius']:.4f}C")

        resumo_geral["modelos"][nome_modelo] = {
            "melhor_epoca": melhor_epoca,
            "metricas_globais": metricas_globais,
            "metricas_por_estacao": metricas_por_estacao,
        }

    with open(RESULTS_DIR / f"resumo_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(resumo_geral, f, indent=2, ensure_ascii=False)

    print(f"\n[Global] Resultados salvos em: {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
