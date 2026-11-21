import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# Configurações Globais
JANELA = 24
N_FEATURES = 3  # temp, hora_sin, hora_cos
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")
print(f"Usando dispositivo: {DEVICE}")

def preparar_dados(df, split=0.8):
    """Cria as janelas deslizantes e converte para Tensores PyTorch."""
    data = df.values
    t_min, t_max = data[:, 0].min(), data[:, 0].max()

    # Normaliza apenas a coluna de temperatura (índice 0)
    data[:, 0] = (data[:, 0] - t_min) / (t_max - t_min)

    X, y = [], []
    for i in range(len(data) - JANELA):
        X.append(data[i:i + JANELA])
        y.append(data[i + JANELA, 0])

    X, y = np.array(X), np.array(y)
    n = int(len(X) * split)

    return (
        torch.tensor(X[:n], dtype=torch.float32),
        torch.tensor(y[:n], dtype=torch.float32),
        torch.tensor(X[n:], dtype=torch.float32),
        torch.tensor(y[n:], dtype=torch.float32),
        (t_min, t_max),
    )


def get_weights(model):
    """Extrai os pesos do modelo como lista de arrays NumPy."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_weights(model, weights):
    """Carrega uma lista de arrays NumPy no modelo."""
    params_dict = zip(model.state_dict().keys(), weights)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)


def train(model, X_train, y_train, epochs=1, lr=0.001, batch_size=64):
    """Executa o treinamento local do modelo."""
    model.to(DEVICE)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    total_loss = 0.0
    for _ in range(epochs):
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            pred = model(X_batch).squeeze()
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, X_test, y_test, batch_size=64):
    """Avalia o modelo no conjunto de teste."""
    model.to(DEVICE)
    model.eval()
    criterion = nn.MSELoss()

    dataset = torch.utils.data.TensorDataset(X_test, y_test)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    total_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            pred = model(X_batch).squeeze()
            total_loss += criterion(pred, y_batch).item()

    mse = total_loss / len(loader)
    rmse = float(np.sqrt(mse))
    return mse, rmse


class DeepLSTM(nn.Module):
    def __init__(self, hidden_size=64, num_layers=2, dropout=0.2):
        super(DeepLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=N_FEATURES,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(1)