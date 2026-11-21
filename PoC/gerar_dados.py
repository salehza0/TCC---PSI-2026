"""
gerar_dados.py
Execute UMA VEZ para gerar os dados sintéticos e salvar em CSV.
  python gerar_dados.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

DADOS_DIR = Path("dados")
DADOS_DIR.mkdir(exist_ok=True)

CLIENTES = {
    0: {"seed": 0,   "ruido": 0.75, "desc": "Estacao limpa"},
    1: {"seed": 42,  "ruido": 6.5,  "desc": "Estacao ruidosa"},
}

def gerar_dados_sinteticos(n_amostras=8760*2, data_inicio='2025-01-01'):
    """Gera dados de temperatura simulando uma estação meteorológica."""
    temp_media_anual = np.random.uniform(18.0, 25.0)
    amp_diaria = np.random.uniform(1.0, 5.0)
    ruido = np.random.uniform(0.1, 0.5)

    datas = pd.date_range(start=data_inicio, periods=n_amostras, freq='h')
    t = np.arange(n_amostras)

    sazonalidade_dia = amp_diaria * np.sin(2 * np.pi * t / 24)
    temp = temp_media_anual + sazonalidade_dia + np.random.normal(0, ruido, n_amostras)

    df = pd.DataFrame({'temp': temp.astype(np.float32)})

    # Encoding Cíclico
    df['hora_sin'] = np.sin(2 * np.pi * datas.hour / 24)
    df['hora_cos'] = np.cos(2 * np.pi * datas.hour / 24)

    return df

for partition_id, cfg in CLIENTES.items():
    np.random.seed(cfg["seed"])
    df = gerar_dados_sinteticos(
        n_amostras=2*8760,
        data_inicio="2025-01-01",
    )
    caminho = DADOS_DIR / f"estacao_{partition_id}.csv"
    df.to_csv(caminho, index=False)
    print(f"[{cfg['desc']}] Salvo em {caminho} | {len(df)} amostras")

print("\nDados gerados com sucesso!")