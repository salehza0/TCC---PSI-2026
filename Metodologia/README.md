# Metodologia: Previsão de Temperatura com Aprendizagem Federada

Implementação dos três paradigmas de treinamento (Local, Centralizado e Federado) usados no trabalho, com as arquiteturas LSTM e TCN e as estratégias de agregação FedAvg, FedMedian, FedYogi e FedAdam.

## Estrutura

```
Metodologia/
├── pyproject.toml          # Configuração do Flower e hiperparâmetros
├── src/
│   ├── task.py             # Modelos (LSTM/TCN), preparação de dados, treino e avaliação
│   ├── client_app.py       # Cliente federado (uma estação = um cliente)
│   ├── server_app.py       # Servidor federado
│   ├── strategies.py       # Estratégias de agregação
│   └── utils/
│       └── experiment_tracker.py   # Salva configurações, métricas e checkpoints
├── treino_global.py        # Baseline Centralizado
└── treino_local.py         # Baseline Local
```

## Instalação

```bash
pip install -e .
```

## Como executar

**Federado**

```bash
flwr run . --stream
```

Os parâmetros (modelo, estratégia, rodadas etc.) são definidos em `pyproject.toml` ou sobrescritos na linha de comando:

```bash
flwr run . --stream --run-config "strategy=fedmedian model=tcn num-server-rounds=50"
```

**Centralizado e Local**

```bash
python -m treino_global
python -m treino_local
```

## Principais parâmetros

| Parâmetro | Descrição | Opções |
|-----------|-----------|--------|
| `model` | Arquitetura | `lstm`, `tcn` |
| `strategy` | Agregação federada | `fedavg`, `fedmedian`, `fedyogi`, `fedadam` |
| `num-server-rounds` | Rodadas de comunicação | inteiro |
| `local-epochs` | Épocas locais por rodada | inteiro |
| `lr` | Taxa de aprendizado | float |

## Fluxo geral

1. Cada estação carrega seus dados, normaliza e monta janelas deslizantes (24 h de entrada, previsão de 1 h).
2. A cada rodada, o servidor envia os pesos globais e cada cliente treina localmente.
3. O servidor agrega os pesos e os clientes avaliam o modelo no próprio conjunto de teste.
4. Os dados brutos nunca saem das estações; apenas pesos e métricas são trocados.

## Métricas

RMSE, MAE e R² em °C, calculados no conjunto de teste de cada estação.

## Resultados

Cada execução gera uma pasta em `results/` (`Federado/`, `Global/` ou `Local/`) com `config.json`, `metricas.json` e o melhor checkpoint.
