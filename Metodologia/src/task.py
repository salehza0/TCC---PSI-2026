import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
from sklearn.metrics import r2_score
# -----------------------------------------------------------------------------
# Configuracoes Globais
# -----------------------------------------------------------------------------

JANELA = 24  # Tamanho da janela deslizante (horas de historico usadas para a previsao)
HORIZONTE = 1 # Quantas horas a frente do fim da janela o modelo deve prever

COLUNA_ALVO = "temperatura_do_ar_bulbo_seco_horaria_c"  # alvo previsto (sempre presente no CSV)
FEATURES_SELECIONADAS = [
    COLUNA_ALVO,
    "hora_sin",
    "hora_cos",
    "dia_ano_sin",
    "dia_ano_cos",
    "umidade_relativa_do_ar_horaria",
    "radiacao_global_kj_m2",
    "pressao_atmosferica_reduzida_nivel_do_mar_aut_mb",
]

N_FEATURES = len(FEATURES_SELECIONADAS)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Task] Usando dispositivo: {DEVICE}")

# -----------------------------------------------------------------------------
# Preparacao de Dados
# -----------------------------------------------------------------------------

def preparar_dados(df: pd.DataFrame, split: float = 0.8,
                    features: Optional[List[str]] = None, verbose: bool = False) -> Tuple:
    """
    Normaliza as colunas numericas (z-score) e converte o DataFrame em janelas
    deslizantes supervisionadas (X, y) prontas para treino.
    """
    features = list(features) if features is not None else list(FEATURES_SELECIONADAS)
    # df = df.loc[df.index >= "2024-01-01"] ## Teste

    data = torch.tensor(df[features].values, dtype=torch.float64)

    # Colunas ciclicas (seno/cosseno de hora e dia) ja estao em [-1, 1], entao nao precisam de normalizacao z-score
    colunas_ciclicas = {"hora_sin", "hora_cos", "dia_ano_sin", "dia_ano_cos"}
    mascara = torch.tensor([c not in colunas_ciclicas for c in features])

    medias, desvios = data.mean(dim=0), data.std(dim=0)
    normalizado = (data - medias) / desvios
    data = torch.where(mascara, normalizado, data)

    idx_alvo = features.index(COLUNA_ALVO)  # COLUNA_ALVO deve estar em features
    t_media, t_desvio = medias[idx_alvo].item(), desvios[idx_alvo].item()
    alvo_norm = normalizado[:, idx_alvo]

    if verbose:
        print(f"[Preparacao] Temperatura normalizada: media={t_media:.1f}C desvio={t_desvio:.1f}C -> z-score")
        for i, coluna in enumerate(features):
            if mascara[i]:
                print(f"[Preparacao] '{coluna}' normalizada: media={medias[i]:.1f} desvio={desvios[i]:.1f} -> z-score")

    entradas = data.numpy()
    alvo = alvo_norm.numpy()

    X, y = [], []
    for i in range(len(entradas) - JANELA - HORIZONTE + 1):
        X.append(entradas[i:i + JANELA])
        y.append(alvo[i + JANELA : i + JANELA + HORIZONTE]) # vetor de HORIZONTE valores

    X, y = np.array(X), np.array(y)

    n = int(len(X) * split)
    X_train, y_train = X[:n], y[:n]
    X_test, y_test = X[n:], y[n:]

    if verbose:
        print(f"[Preparacao] Sequencias criadas: X.shape={X.shape}, y.shape={y.shape}")
        print(f"[Preparacao] Split {split:.0%}: "
              f"treino={len(X_train)} | teste={len(X_test)}")

    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
        (t_media, t_desvio),
    )


# -----------------------------------------------------------------------------
# Manipulacao de Pesos (Flower)
# -----------------------------------------------------------------------------

def get_weights(model: nn.Module) -> list:
    """Extrai os pesos do modelo como lista de arrays NumPy (formato Flower)."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_weights(model: nn.Module, weights: list) -> None:
    """Carrega uma lista de arrays NumPy (formato Flower) no modelo."""
    params_dict = zip(model.state_dict().keys(), weights)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)


# -----------------------------------------------------------------------------
# Treinamento e Avaliacao
# -----------------------------------------------------------------------------

def train(model: nn.Module, X_train: torch.Tensor, y_train: torch.Tensor,
          epochs: int = 1, lr: float = 0.001, batch_size: int = 64) -> float:
    """
    Treina o modelo com Adam + MSELoss.

    Args:
        model: modelo a treinar (DeepLSTM ou TCN)
        X_train: (n_amostras, JANELA, N_FEATURES)
        y_train: (n_amostras, HORIZONTE)
        epochs: numero de epocas
        lr: taxa de aprendizado
        batch_size: tamanho do batch

    Returns:
        Loss medio da ultima epoca.
    """
    model.to(DEVICE)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    total_loss = 0.0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            pred = model(X_batch)  # (batch, HORIZONTE) - sem squeeze
            loss = criterion(pred, y_batch)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        total_loss = epoch_loss / len(loader)
        if epochs > 1:
            print(f"  [Treino] Epoca {epoch + 1}/{epochs}: Loss={total_loss:.6f}")

    return total_loss

def evaluate(model: nn.Module, X_test: torch.Tensor, y_test: torch.Tensor,
             batch_size: int = 64,
             escala: Optional[Tuple[float, float]] = None) -> dict:
    """
    Avalia o modelo em MSE/RMSE/MAE agregados (media sobre todas as amostras
    e todos os passos do horizonte), normalizado e desnormalizado (C).
    """
    model.to(DEVICE)
    model.eval()

    dataset = torch.utils.data.TensorDataset(X_test, y_test)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    soma_erro_quadrado, soma_erro_absoluto, n_total = 0.0, 0.0, 0
    preds, trues = [], []   # novo: acumula para o R2
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            pred = model(X_batch)  # (batch, HORIZONTE)

            # soma sobre batch x horizonte; divide pelo total real no final
            soma_erro_quadrado += ((pred - y_batch) ** 2).sum().item()
            soma_erro_absoluto += (pred - y_batch).abs().sum().item()
            n_total += y_batch.numel()  # batch_size * HORIZONTE

            preds.append(pred.detach().cpu().numpy())     
            trues.append(y_batch.detach().cpu().numpy())   

    mse_norm = soma_erro_quadrado / n_total
    mae_norm = soma_erro_absoluto / n_total
    rmse_norm = float(np.sqrt(mse_norm))

    #R2 sobre todas as amostras e todos os passos do horizonte
    y_pred_all = np.concatenate(preds).ravel()
    y_true_all = np.concatenate(trues).ravel()
    r2 = float(r2_score(y_true_all, y_pred_all))

    if escala is not None:
        t_media, t_desvio = escala
        mse_celsius = mse_norm * (t_desvio ** 2)
        rmse_celsius = float(np.sqrt(mse_celsius))
        mae_celsius = mae_norm * t_desvio
    else:
        mse_celsius, rmse_celsius, mae_celsius = mse_norm, rmse_norm, mae_norm

    return {
        "mse_norm": mse_norm,
        "rmse_norm": rmse_norm,
        "mae_norm": mae_norm,
        "mse_celsius": mse_celsius,
        "rmse_celsius": rmse_celsius,
        "mae_celsius": mae_celsius,
        "r2": r2,
    }

# -----------------------------------------------------------------------------
# Modelo Deep LSTM
# -----------------------------------------------------------------------------

class DeepLSTM(nn.Module):
    """LSTM empilhada + cabeca densa, prevendo as proximas HORIZONTE horas de temperatura."""

    def __init__(self, hidden_size: int = 64, num_layers: int = 1,
                 dropout: float = 0.2, n_features: int = N_FEATURES):

        super(DeepLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, HORIZONTE),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, (h_n, c_n) = self.lstm(x)
        return self.fc(out[:, -1, :])  # ultimo passo temporal


# -----------------------------------------------------------------------------
# Modelo TCN
# -----------------------------------------------------------------------------
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    """TemporalConvNet + cabeca densa, prevendo as proximas HORIZONTE horas de temperatura."""

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Bloco residual de duas convolucoes causais dilatadas (unidade basica da TCN)."""

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """Pilha de TemporalBlocks com dilatacao crescente (2^i), formando o backbone da TCN."""

    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    """TemporalConvNet + cabeca densa, prevendo a temperatura da proxima hora."""

    def __init__(self, num_channels: Optional[List[int]] = None, kernel_size: int = 3,
                 dropout: float = 0.2, n_features: int = N_FEATURES):
        super(TCN, self).__init__()

        if num_channels is None:
            num_channels = [32, 32, 16]

        self.tcn = TemporalConvNet(
            num_inputs=n_features,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        self.fc = nn.Sequential(
            nn.Linear(num_channels[-1], 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, HORIZONTE),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (batch, JANELA, features) -> (batch, canais, sequencia), formato do Conv1d
        out = self.tcn(x)
        return self.fc(out[:, :, -1])  # ultimo passo temporal


# -----------------------------------------------------------------------------
# Factory de Modelos
# -----------------------------------------------------------------------------

MODELOS_DISPONIVEIS = {
    "lstm": DeepLSTM,
    "tcn": TCN,
}


def criar_modelo(model_name: str = "lstm", **kwargs) -> nn.Module:
    """Instancia o modelo (DeepLSTM ou TCN) a partir do nome, configuravel via pyproject.toml."""
    model_name = model_name.lower().strip()

    if model_name not in MODELOS_DISPONIVEIS:
        raise ValueError(
            f"Modelo '{model_name}' nao reconhecido. "
            f"Disponiveis: {list(MODELOS_DISPONIVEIS.keys())}"
        )

    return MODELOS_DISPONIVEIS[model_name](**kwargs)
