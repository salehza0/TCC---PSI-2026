# TCC — Aprendizado Federado para Previsão de Temperatura

Projeto de prova de conceito utilizando **Flower (flwr)** + **PyTorch** para treinar
um modelo LSTM de forma federada, simulando 2 estações meteorológicas com dados
sintéticos e climas distintos.

---

## Estrutura do Projeto

```
prova_conceito/
├── src/
│   ├── __init__.py
│   ├── task.py         # Modelo DeepLSTM, geração de dados, treino e avaliação
│   ├── client_app.py   # Lógica do cliente federado (estação meteorológica)
│   └── server_app.py   # Lógica do servidor (agregação FedAvg, FedMedian, FedYogi ou FedAdam)
├── pyproject.toml      # Configurações do projeto e do Flower
└── README.md
```

---

## Instalação

```bash
conda activate flower-env
cd C:\TCC\prova_conceito
pip install -e .
```

---

## Como Executar

### Rodando com FedAvg (padrão)

```bash
flwr run . --stream
```


Depois rode novamente:
```bash
flwr run . --stream
```

### Sobrescrevendo parâmetros direto no terminal (sem editar o arquivo)

```bash
# FedAvg com 5 rodadas e 5 épocas locais
flwr run . --stream --run-config "num-server-rounds=5 local-epochs=5 strategy=fedavg"
```

---

## Parâmetros Configuráveis (`pyproject.toml`)

| Parâmetro           | Descrição                                      | Padrão    |
|---------------------|------------------------------------------------|-----------|
| `num-server-rounds` | Número de rodadas de comunicação federada      | `10`      |
| `strategy`          | Estratégia de agregação: `fedavg`, `fedmedian`, `fedyogi` ou `fedadam` | `fedavg` |
| `local-epochs`      | Épocas de treino local por rodada              | `3`       |
| `batch-size`        | Tamanho do batch no treino local               | `64`      |
| `lr`                | Taxa de aprendizado                            | `0.001`   |
| `fraction-fit`      | Fração de clientes que treinam por rodada      | `1.0`     |
| `fraction-evaluate` | Fração de clientes que avaliam por rodada      | `1.0`     |

---

## Métricas Monitoradas

- **train_loss**: MSE do treino local em cada cliente
- **mse**: Mean Squared Error agregado da federação
- **rmse**: Raiz do MSE (em °C) — métrica principal do TCC

---

## Fluxo Federado

```
Rodada 1..N:
  [Servidor] envia pesos globais
       ↓
  [Cliente 0]  treina localmente → devolve pesos + métricas
  [Cliente 1]  treina localmente → devolve pesos + métricas
       ↓
  [Servidor] agrega pesos (FedAvg, FedMedian, FedYogi ou FedAdam)
  [Servidor] calcula RMSE médio ponderado
```
