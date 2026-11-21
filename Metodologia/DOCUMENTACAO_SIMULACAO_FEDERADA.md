# Documentação da Simulação Federada

Este documento explica, passo a passo, tudo que acontece quando você roda
`flwr run . --stream` neste projeto: quais arquivos são executados, em que
ordem, o que cada função faz e como eles se conectam. O objetivo é servir de
material de apoio para escrever a metodologia do TCC (Capítulo 5) com
segurança sobre o que o código realmente faz.

---

## 1. Visão geral do fluxo

```
flwr run . --stream
        │
        ├─► lê pyproject.toml (seção [tool.flwr.app])
        │     └─ descobre onde estão o ServerApp e o ClientApp
        │
        ├─► inicia o SERVIDOR (src/server_app.py -> server_fn)
        │     └─ cria o modelo global, a estratégia de agregação e o tracker
        │
        └─► inicia N CLIENTES simulados via Ray (src/client_app.py -> client_fn)
              └─ um por estação meteorológica (5 no total)

Depois disso, o Flower assume o controle e repete, por `num-server-rounds`
rodadas:

  1. Servidor manda os pesos globais atuais para os clientes selecionados
  2. Cada cliente treina localmente (fit) e devolve pesos + métricas de treino
  3. Servidor agrega os pesos (estratégia escolhida: FedAvg/FedMedian/FedYogi/FedAdam)
  4. Servidor manda os pesos agregados para avaliação (evaluate)
  5. Cada cliente avalia no seu conjunto de teste local e devolve métricas
  6. Servidor agrega as métricas de avaliação e registra a rodada
```

Os cinco arquivos envolvidos e a responsabilidade de cada um:

| Arquivo | Responsabilidade |
|---|---|
| `pyproject.toml` | Configuração central: hiperparâmetros, modelo, estratégia |
| `src/task.py` | Definição dos modelos (LSTM/TCN), preparação de dados, treino e avaliação |
| `src/client_app.py` | Lógica de cada cliente (estação meteorológica) |
| `src/strategies.py` | As 4 estratégias de agregação federada |
| `src/server_app.py` | Monta o servidor: modelo global, estratégia, agregação de métricas |
| `src/utils/experiment_tracker.py` | Salva config/métricas/checkpoints de cada execução em disco |

---

## 2. `pyproject.toml` — configuração central

Este arquivo não tem lógica, mas é o ponto de partida: ele diz ao Flower
onde estão o `ServerApp` e o `ClientApp` e define todos os hiperparâmetros
que os outros arquivos vão ler via `context.run_config`.

```toml
[tool.flwr.app.components]
serverapp = "src.server_app:app"
clientapp = "src.client_app:app"

[tool.flwr.app.config]
num-server-rounds = 30      # quantas rodadas de comunicação servidor <-> clientes
strategy = "fedavg"          # fedavg | fedmedian | fedyogi | fedadam
model = "lstm"                # lstm | tcn
local-epochs = 3              # épocas de treino local por rodada, em cada cliente
batch-size = 64
lr = 0.001                    # taxa de aprendizado do Adam local
fraction-fit = 1.0             # fração de clientes usados no treino a cada rodada
fraction-evaluate = 1.0        # fração de clientes usados na avaliação a cada rodada

# Só usados quando strategy = "fedadam" ou "fedyogi"
eta = 0.01        # taxa de aprendizado do servidor
eta-l = 0.1        # taxa de aprendizado local (specific do FedOpt)
beta-1 = 0.9       # decaimento do 1o momento (tipo Adam)
beta-2 = 0.99      # decaimento do 2o momento
tau = 1e-3         # estabilidade numérica (denominador nunca chega a zero)
```

Nenhum desses valores está hardcoded em nenhum `.py` — tudo é lido de volta
por `context.run_config.get(...)` dentro de `client_app.py` e `server_app.py`.
Isso significa que **trocar modelo ou estratégia de agregação é só mudar
uma linha aqui**, sem tocar em código.

---

## 3. `src/task.py` — modelo, dados, treino e avaliação

Este é o único módulo que não sabe nada sobre Flower ou aprendizado
federado — ele só define blocos reutilizáveis que tanto o cliente quanto o
servidor importam.

### 3.1 Configuração global (topo do arquivo)

```python
JANELA = 24        # quantas horas de histórico entram como entrada (X)
HORIZONTE = 1      # quantas horas à frente do fim da janela o modelo prevê

COLUNA_ALVO = "temperatura_do_ar_bulbo_seco_horaria_c"
FEATURES_SELECIONADAS = [ ... ]   # quais colunas do CSV entram como entrada
N_FEATURES = len(FEATURES_SELECIONADAS)   # calculado automaticamente
```

`FEATURES_SELECIONADAS` é a lista que você edita pra escolher quais
variáveis (temperatura, umidade, radiação, pressão, seno/cosseno de
hora/dia) entram no modelo — remover uma linha daqui já reduz `N_FEATURES`
e a entrada de todos os modelos, sem mexer em mais nada.

### 3.2 `preparar_dados(df, split, features, verbose)`

Recebe o DataFrame bruto de uma estação e devolve tensores prontos pra
treino. Passos internos:

1. Seleciona só as colunas de `features` do DataFrame.
2. Normaliza por **z-score** (`(x - média) / desvio`) todas as colunas
   exceto as cíclicas (`hora_sin`, `hora_cos`, `dia_ano_sin`, `dia_ano_cos`),
   que já estão em `[-1, 1]` e não precisam de normalização.
3. Guarda separadamente a média/desvio da coluna-alvo (`t_media`,
   `t_desvio`) — isso é o que depois permite desnormalizar o erro de volta
   para °C.
4. Monta as **janelas deslizantes**: para cada instante `i`, `X[i]` é o
   bloco de `JANELA=24` horas anteriores e `y[i]` é a temperatura
   `HORIZONTE` horas depois do fim dessa janela.
5. Faz o split treino/teste **cronológico** (sem embaralhar, já que é série
   temporal): os primeiros `split%` (padrão 80%) viram treino, o resto vira
   teste.
6. Retorna `(X_train, y_train, X_test, y_test, (t_media, t_desvio))`.

Cada uma das 5 estações chama essa função de forma independente com seu
próprio CSV — não há vazamento de dados entre estações nessa etapa.

### 3.3 `get_weights` / `set_weights`

Convertem entre o formato nativo do PyTorch (`state_dict`) e o formato que o
Flower espera para transmitir pesos pela rede (lista de arrays NumPy). São
usadas toda vez que o cliente recebe pesos do servidor (`set_weights`) ou
devolve pesos atualizados (`get_weights`).

### 3.4 `train(model, X_train, y_train, epochs, lr, batch_size)`

Loop de treino padrão do PyTorch: otimizador Adam, `MSELoss`, itera por
`epochs` épocas sobre os dados em mini-batches. Retorna a loss média da
última época. É chamada uma vez por cliente, por rodada (dentro de
`fit()` em `client_app.py`).

### 3.5 `evaluate(model, X_test, y_test, batch_size, escala)`

Roda o modelo em modo de avaliação (sem gradiente) e calcula MSE, RMSE e
MAE, tanto na escala normalizada quanto desnormalizada:

```python
mse_celsius = mse_norm * (t_desvio ** 2)      # RMSE cresce com o quadrado
mae_celsius = mae_norm * t_desvio             # MAE é linear
```

`escala` é a tupla `(t_media, t_desvio)` devolvida por `preparar_dados` —
é o elo entre a normalização feita na hora de preparar os dados e a
métrica em °C reportada no final. Chamada tanto pelo cliente (avaliação
local) quanto pelo `treino_global.py` (baseline centralizado).

### 3.6 `DeepLSTM` e `TCN`

As duas arquiteturas de modelo (detalhadas na Seção 5.2.1 do TCC). Ambas
recebem `n_features=N_FEATURES` e produzem uma única saída escalar (a
temperatura prevista). `criar_modelo(model_name)` é a fábrica que escolhe
entre as duas a partir da string `"lstm"` ou `"tcn"` vinda do
`pyproject.toml`.

---

## 4. `src/client_app.py` — cada estação como um cliente federado

### 4.1 `ARQUIVOS_ESTACOES` e `client_fn(context)`

O Flower cria clientes numerados por `partition_id` (0 a 4). Esse índice é
usado só para escolher qual arquivo CSV (`ARQUIVOS_ESTACOES[partition_id]`)
aquele cliente representa — a lista fixa garante que a estação A701 seja
sempre `partition_id=0`, A706 sempre `partition_id=1`, etc., de forma
consistente entre rodadas.

Dentro de `client_fn`, cada cliente:

1. Lê `local-epochs`, `batch-size` e `model` do `context.run_config`
   (vindo do `pyproject.toml`).
2. Verifica o cache `_cache_clientes` — se esse `partition_id` já carregou o
   CSV antes **nesse processo** (ator do Ray), reaproveita os tensores já
   prontos em vez de reler o disco e reprocessar tudo de novo. É por isso
   que o log `"CSV carregado"` só aparece uma vez por estação por worker,
   não a cada rodada.
3. Instancia `EstacaoMeteorologicaClient` com os dados e devolve ao Flower.

### 4.2 `EstacaoMeteorologicaClient` (implementa `NumPyClient`)

Duas funções que o Flower chama automaticamente a cada rodada:

- **`fit(parameters, config)`**: recebe os pesos globais atuais
  (`parameters`), carrega-os no modelo local (`set_weights`), treina por
  `local_epochs` épocas (`train(...)`) e devolve `(pesos_atualizados,
  n_amostras_treino, métricas)`. `n_amostras_treino` é usado pela
  estratégia de agregação para ponderar a média dos pesos (estações com
  mais dados pesam mais no `FedAvg`).
- **`evaluate(parameters, config)`**: recebe os pesos globais agregados,
  avalia no conjunto de teste local (`evaluate(...)` de `task.py`) e
  devolve `(mse_norm, n_amostras_teste, métricas)` — incluindo o código da
  estação (`"estacao": self.codigo_estacao`), que é o que permite ao
  servidor depois imprimir o resultado por estação em vez de só o
  agregado.

Importante: **os dados brutos (`X_train`, `y_train`, etc.) nunca saem
dessa classe** — só os pesos do modelo e os dicionários de métricas
trafegam de volta para o servidor. Isso é o que caracteriza o esquema como
aprendizado federado (Seção 2.3 do TCC) e não treinamento centralizado.

---

## 5. `src/strategies.py` — as 4 estratégias de agregação

### 5.1 Por que existe um "mixin"

FedAvg, FedMedian, FedYogi e FedAdam do Flower nativo já implementam a
matemática de agregação (Seção 2.3.2 do TCC). O que esse arquivo adiciona,
em cima disso, é comportamento **igual para as quatro estratégias**:
imprimir o resultado de cada cliente por rodada e salvar pesos/métricas no
tracker. Em vez de reescrever esse código 4 vezes, ele fica uma vez só em
`_SalvamentoMixin` / `_AggregateComSalvamento`, e cada estratégia concreta
(`FedAvgComSalvamento`, etc.) só herda desse mixin **e** da classe nativa
correspondente do Flower.

### 5.2 `aggregate_fit(server_round, results, failures)`

Chamado pelo servidor depois que todos os clientes selecionados terminam o
`fit`. Faz três coisas, nessa ordem:

1. `_imprimir_fit_clientes`: imprime uma linha por estação com a loss de
   treino daquela rodada.
2. `super().aggregate_fit(...)`: aqui entra a matemática de verdade — média
   ponderada (FedAvg), mediana (FedMedian), ou a atualização adaptativa tipo
   Adam/Yogi (FedAdam/FedYogi) — dependendo de qual classe nativa está por
   trás.
3. `_salvar_pesos_fit`: guarda os pesos agregados resultantes (pro tracker
   salvar como checkpoint depois).

### 5.3 `aggregate_evaluate(server_round, results, failures)`

Chamado depois que todos os clientes terminam a `evaluate`. Delega a
agregação das métricas para `super()` (que por sua vez chama a função
`weighted_average_metrics` definida em `server_app.py` — ver Seção 6), e
depois `_processar_resultado` imprime o RMSE/MAE de cada estação e manda o
resultado agregado da rodada para o `tracker.registrar_rodada`.

### 5.4 As 4 estratégias concretas

```python
class FedAvgComSalvamento(_AggregateComSalvamento, FedAvg): ...
class FedMedianComSalvamento(_AggregateComSalvamento, FedMedian): ...
class FedYogiComSalvamento(_AggregateComSalvamento, FedYogi): ...
class FedAdamComSalvamento(_AggregateComSalvamento, FedAdam): ...
```

Cada uma só chama o `__init__` da classe nativa do Flower (passando os
hiperparâmetros vindos do `pyproject.toml`) e inicializa o mixin de
salvamento. `create_strategy(strategy_name, ...)` é a fábrica que escolhe
qual dessas classes instanciar a partir da string do `pyproject.toml`
(chamada de dentro de `server_app.py`).

---

## 6. `src/server_app.py` — monta o servidor

### 6.1 `weighted_average_metrics(metrics)`

É a função de agregação de métricas de avaliação (passada ao Flower como
`evaluate_metrics_aggregation_fn`). Recebe uma lista de
`(n_amostras_daquele_cliente, dict_de_metricas)` — uma entrada por
estação — e calcula a média ponderada pelo número de amostras de cada
estação (a mesma lógica da Equação 2.9 do TCC, mas aplicada às métricas em
vez de aos pesos). Também imprime a linha `"[Servidor] Metricas
agregadas | ..."` que aparece uma vez por rodada.

### 6.2 `server_fn(context)`

Função de fábrica chamada uma única vez, no início da simulação. Nessa
ordem:

1. Lê os hiperparâmetros do `pyproject.toml` (`num-server-rounds`,
   `strategy`, `model`, `fraction-fit`, `fraction-evaluate`).
2. Cria o **modelo global inicial** (`criar_modelo(model_nome)`) e converte
   seus pesos para o formato Flower (`ndarrays_to_parameters`) — é esse
   conjunto de pesos aleatórios que é distribuído a todos os clientes na
   primeira rodada.
3. Inicializa o `ExperimentTracker` (Seção 7), que já cria a pasta de
   resultados daquela execução e salva a configuração usada.
4. Monta `kwargs_comuns`: os argumentos que toda estratégia do Flower
   aceita (`fraction_fit`, `min_fit_clients=2`, `initial_parameters`,
   `evaluate_metrics_aggregation_fn=weighted_average_metrics`, etc.). O
   `min_fit_clients=2` é só uma salvaguarda — com as 5 estações sempre
   disponíveis, nunca chega a ser acionado.
5. Se a estratégia escolhida for `fedadam` ou `fedyogi`, adiciona também os
   hiperparâmetros específicos do FedOpt (`eta`, `eta_l`, `beta_1`,
   `beta_2`, `tau`).
6. Chama `create_strategy(...)` (de `strategies.py`) para instanciar a
   estratégia concreta.
7. Devolve `ServerAppComponents(strategy=estrategia, config=ServerConfig(num_rounds=num_rounds))`
   — é esse retorno que o Flower usa para efetivamente rodar a simulação.

---

## 7. `src/utils/experiment_tracker.py` — persistência de resultados

Não faz parte da lógica federada em si (o Flower funcionaria sem ele), mas
é o que transforma cada execução em um experimento reproduzível e
analisável depois:

- **`__init__`**: cria uma pasta com timestamp
  (`results/Federado/<strategy>_<model>_<timestamp>/`) para aquela
  execução específica.
- **`salvar_config`**: grava um `config.json` com todos os hiperparâmetros
  usados (chamado uma vez, no início, por `server_fn`).
- **`capturar_pesos_agregados`**: guarda os pesos agregados da rodada atual
  em memória (chamado por `_salvar_pesos_fit` em `strategies.py`).
- **`registrar_rodada`**: chamado a cada rodada (por
  `_processar_resultado` em `strategies.py`) com o RMSE/MSE/MAE agregados e
  o dicionário `clientes_info` (métricas por estação). Se o RMSE dessa
  rodada for o melhor até agora, salva os pesos em
  `checkpoints/melhores_pesos.pt` e atualiza `melhor_rmse`/`melhor_rodada`.
  Reescreve `metricas.json` a cada rodada com o histórico completo.
- **`gerar_resumo_final`**: imprime um resumo (rodadas totais, melhor RMSE,
  % de melhoria) ao final da simulação.

---

## 8. Passo a passo de uma rodada completa (rastreio ponta a ponta)

Juntando tudo, isto é o que acontece fisicamente em **uma rodada** de
comunicação (por exemplo, a Rodada 5), assumindo `strategy="fedavg"`:

1. **Flower (core)** decide, com base em `fraction_fit=1.0`, que os 5
   clientes vão participar do treino desta rodada.
2. Para cada um dos 5 clientes (em paralelo, via atores do Ray):
   - `client_fn` é chamado (ou reaproveitado, se o ator já rodou antes) →
     devolve um `EstacaoMeteorologicaClient` com os dados daquela estação
     já carregados/cacheados.
   - `.fit(pesos_globais_atuais, config)` é chamado → treina localmente por
     `local-epochs` épocas → devolve `(pesos_novos, n_amostras, {"estacao":
     ..., "train_loss": ...})`.
3. O core do Flower junta os 5 resultados em uma lista `results` e chama
   `FedAvgComSalvamento.aggregate_fit(5, results, failures=[])`.
   - Isso imprime a linha de loss por estação (`_imprimir_fit_clientes`).
   - Isso delega a média ponderada dos 5 conjuntos de pesos para o
     `FedAvg` nativo do Flower (`super().aggregate_fit`).
   - Isso guarda os pesos agregados resultantes (`_salvar_pesos_fit`, que
     por sua vez avisa o tracker).
4. Com base em `fraction_evaluate=1.0`, os mesmos 5 clientes são
   selecionados para avaliação, recebendo os pesos **agregados** (o novo
   modelo global) e chamando `.evaluate(pesos_agregados, config)` cada um
   → devolve `(mse_norm, n_amostras_teste, {"estacao": ..., "rmse": ...,
   "mae": ..., ...})`.
5. O Flower agrega essas métricas chamando `weighted_average_metrics`
   (média ponderada pelo número de amostras de teste de cada estação) e
   passa o resultado para
   `FedAvgComSalvamento.aggregate_evaluate(5, results, failures=[])`.
6. `_processar_resultado` imprime o RMSE/MAE por estação e chama
   `tracker.registrar_rodada(5, rmse_agregado, mse_agregado, clientes_info,
   mae_global=mae_agregado)` — se esse for o melhor RMSE até agora, os
   pesos do modelo global são salvos em `melhores_pesos.pt`.
7. O Flower avança para a Rodada 6, repetindo o processo com os novos
   pesos globais como ponto de partida.

Depois de `num-server-rounds` rodadas, a simulação termina e o
`ExperimentTracker` já tem, em disco, o histórico completo (`metricas.json`),
a configuração usada (`config.json`) e o melhor checkpoint
(`melhores_pesos.pt`) daquela execução — prontos para a análise
comparativa entre modelos e estratégias no Capítulo 6 do TCC.

---

## 9. Onde cada peça aparece no TCC

Para referência rápida ao escrever a Metodologia (Seção 5.2):

- **5.2.1 (Arquiteturas)** → `task.py`, classes `DeepLSTM` e `TCN`.
- **5.2.2 (Ambiente Federado)** → `client_app.py`, `server_app.py`,
  `strategies.py` (as 4 estratégias e o mixin de reuso de código).
- **5.2.3 (Parâmetros e Protocolo)** → `pyproject.toml`
  (`[tool.flwr.app.config]`), `preparar_dados` em `task.py` (normalização
  z-score, janela de 24h, split 80/20), `evaluate` em `task.py` (RMSE/MAE
  desnormalizados).
- **5.2.4 (Organização do Código)** → estrutura de arquivos como um todo,
  descrita nas Seções 3–7 deste documento.

---

## 10. Baselines de comparação: `treino_global.py` e `treino_local.py`

Esses dois scripts **não fazem parte da simulação federada** — não usam
Flower, não têm cliente/servidor, não trocam pesos entre estações. Eles
existem para gerar os outros dois pontos de comparação da Figura 1 do TCC
(Modelo Centralizado e Modelos Locais), rodados como scripts Python comuns
(`python -m treino_global` / `python -m treino_local`), e reaproveitam
diretamente `preparar_dados`, `train`, `evaluate`, `DeepLSTM` e `TCN` de
`task.py` — por isso os resultados são comparáveis com os do modelo
federado: mesma preparação de dados, mesma arquitetura, mesma métrica.

### 10.1 `treino_global.py` — Modelo Centralizado

Simula o que aconteceria se todos os dados de todas as estações fossem
enviados para um único servidor e treinados juntos, sem federação.

- **`carregar_estacoes(dados_dir)`**: lê todos os CSVs da pasta `dados/`
  (ignorando `dados_gerais.csv`) e devolve um dict `{código: DataFrame}` —
  uma entrada por estação.
- **`preparar_dados_globais(estacoes, split)`**: chama `preparar_dados` de
  `task.py` **separadamente para cada estação** (cada uma com sua própria
  normalização z-score e seu próprio split cronológico 80/20), e só depois
  concatena (`torch.cat`) os tensores de treino de todas as estações num
  único `X_train`/`y_train` — e o mesmo para o teste. Isso evita que dados
  de uma estação vazem para a normalização de outra, mas junta tudo num
  único conjunto de treino, que é a definição de "centralizado". Também
  devolve `X_tests`/`y_tests`/`escalas` **por estação**, para poder medir o
  desempenho do modelo global também individualmente em cada uma depois.
- **`treinar_com_checkpoint(model_class, X_train, y_train, X_test, y_test, epochs, lr, batch_size)`**:
  treina época a época (chamando `train(...)` com `epochs=1` em loop, em
  vez de passar `epochs=N` de uma vez) para poder avaliar
  (`evaluate(...)`) o modelo no conjunto de teste **depois de cada
  época** e guardar os pesos (`copy.deepcopy(modelo.state_dict())`) da
  época com menor `rmse_norm`. Ou seja, o checkpoint final não é
  necessariamente o da última época — é o de melhor generalização vista
  durante o treino (proteção manual contra overfitting).
- **`main()`**: parseia argumentos de linha de comando (`--modelos`,
  `--epochs`, `--lr`, `--batch-size`, `--split`), roda
  `treinar_com_checkpoint` para cada modelo pedido (LSTM e/ou TCN), calcula
  as métricas finais tanto agregadas (`metricas_globais`, no conjunto de
  teste combinado) quanto por estação (`metricas_por_estacao`, usando os
  `X_tests`/`escalas` individuais salvos antes), e grava tudo em
  `results/Global/<modelo>_<timestamp>/` (`melhor.pt` com os pesos +
  métricas, `metricas.json` com o histórico completo por época) e um
  `resumo_<timestamp>.json` consolidado no nível acima.

### 10.2 `treino_local.py` — Modelos Locais (isolados)

Simula o oposto do centralizado: cada estação treina seu próprio modelo,
do zero, sem nenhuma troca de informação com as outras — é o "modelo
puramente local" discutido nas Seções 1.2 e 4.1 do TCC.

- Reaproveita a mesma `carregar_estacoes(dados_dir)` de `treino_global.py`.
- **`treinar_com_checkpoint(model_class, df, epochs, lr, batch_size, split)`**:
  praticamente igual à versão do `treino_global.py`, mas com uma diferença
  importante — aqui `preparar_dados` é chamado **uma única vez, dentro da
  própria função**, para o `df` de uma única estação (em vez de já receber
  tensores prontos de fora). Além disso, o critério de melhor época usa
  `rmse_celsius` (já desnormalizado) em vez de `rmse_norm`, porque aqui não
  há necessidade de comparar entre estações com escalas diferentes — cada
  estação tem seu próprio modelo e sua própria escala de temperatura.
- **`main()`**: itera em **duas camadas de loop** — para cada estação,
  para cada modelo pedido —, chamando `treinar_com_checkpoint` de forma
  totalmente independente por combinação (ou seja, 5 estações × 2 modelos
  = 10 treinos isolados do zero). Salva os resultados em
  `results/Local/<estacao>/<modelo>_<timestamp>/` (uma subpasta por
  estação, já que aqui não existe um "resultado agregado" — cada estação é
  seu próprio experimento) e um `resumo_<timestamp>.json` com o resultado
  de todas as combinações estação × modelo.

### 10.3 Como esses dois se encaixam com o Flower

| | `treino_local.py` | `treino_global.py` | Simulação Federada (`flwr run`) |
|---|---|---|---|
| Dados de treino | 1 estação por modelo | todas as estações juntas | cada estação treina só com os próprios dados |
| Compartilhamento entre estações | nenhum | total (dados brutos concatenados) | só pesos/métricas, nunca dados brutos |
| Nº de modelos treinados | 1 por estação (5×2) | 1 único modelo global (2) | 1 modelo global, mas nunca vê dados brutos de nenhuma estação |
| Onde entra no TCC | ponta "Modelos Locais" da Fig. 1 | ponta "Modelo Centralizado" da Fig. 1 | ponta "Modelo Federado" da Fig. 1 |

Os três usam exatamente a mesma `preparar_dados` (mesma normalização
z-score, mesma janela de 24h, mesmo split 80/20 cronológico) e as mesmas
arquiteturas `DeepLSTM`/`TCN` de `task.py` — a única variável entre os três
paradigmas é **como o treinamento é distribuído entre as estações**, o que
é exatamente o que o Capítulo 6 (Resultados) precisa comparar.
