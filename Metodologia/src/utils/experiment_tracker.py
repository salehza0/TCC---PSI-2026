"""
experiment_tracker.py - Sistema robusto de rastreamento de experimentos

Gerencia:
  - Criacao de diretorios com timestamp
  - Salvamento de config, metricas e pesos
  - Versionamento automatico
  - Analise comparativa pos-treino
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import numpy as np
import torch

class ExperimentTracker:
    def __init__(self, base_results_dir: Path, strategy_name: str, model_name: str = "lstm"):
        """
        Args:
            base_results_dir: diretorio base para resultados (ex: Path("results"))
            strategy_name: nome da estrategia (fedavg, fedmedian, etc)
            model_name: nome do modelo (lstm ou tcn), usado para recriar o
                modelo ao salvar checkpoints
        """
        self.strategy_name = strategy_name
        self.model_name = model_name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = f"{strategy_name}_{model_name}_{timestamp}"
        self.experiment_dir = base_results_dir / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.config_file = self.experiment_dir / "config.json"
        self.metricas_file = self.experiment_dir / "metricas.json"
        self.clientes_file = self.experiment_dir / "metricas_clientes.csv"
        self.pesos_dir = self.experiment_dir / "checkpoints"
        self.pesos_dir.mkdir(parents=True, exist_ok=True)
        
        self.historico: List[Dict[str, Any]] = []
        self.melhor_rmse = float("inf")
        self.melhor_rodada = -1
        self._pesos_rodada_atual = None
        
        print(f"[ExperimentTracker] Criado: {self.experiment_dir.resolve()}")
    
    def salvar_config(self, config: Dict[str, Any]):
        """Salva config inicial do experimento (hiperparametros)."""
        config["timestamp"] = datetime.now().isoformat()
        config["experiment_name"] = self.experiment_name
        
        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"[ExperimentTracker] Config salvo: {self.config_file.name}")
    
    def capturar_pesos_agregados(self, aggregated_parameters):
        """Captura pesos apos aggregate_fit (mesmo que server_app.py)."""
        if aggregated_parameters is not None:
            from flwr.common import parameters_to_ndarrays
            self._pesos_rodada_atual = parameters_to_ndarrays(aggregated_parameters)
            print(f"[ExperimentTracker] Pesos capturados | {len(self._pesos_rodada_atual)} arrays")
        else:
            print("[ExperimentTracker] AVISO: aggregated_parameters e None!")
    
    def registrar_rodada(self, server_round: int, rmse_global: float,
                        mse_global: float, clientes_info: Dict[str, Dict],
                        mae_global: float = float("nan")):
        """Registra metricas de uma rodada."""
        entrada = {
            "rodada": server_round,
            "rmse_global": float(rmse_global),
            "mse_global": float(mse_global),
            "mae_global": float(mae_global),
            "melhor_ate_agora": False,
            "timestamp": datetime.now().isoformat(),
            "clientes": clientes_info,
        }
        
        # Atualiza melhor RMSE e salva pesos
        if rmse_global < self.melhor_rmse and self._pesos_rodada_atual is not None:
            self.melhor_rmse = rmse_global
            self.melhor_rodada = server_round
            entrada["melhor_ate_agora"] = True
            
            self._salvar_pesos_checkpoint(server_round, rmse_global, is_best=True)
            print(f"[ExperimentTracker] Melhor RMSE encontrado! "
                  f"Rodada {server_round} | RMSE: {rmse_global:.4f}C")
        
        self.historico.append(entrada)
        self._salvar_metricas_json()
    
    def salvar_checkpoint_periodico(self, server_round: int, rmse_global: float):
        """Salva checkpoint a cada N rodadas (util para analise de convergencia)."""
        if self._pesos_rodada_atual is not None:
            self._salvar_pesos_checkpoint(server_round, rmse_global, is_best=False)
    
    def _salvar_pesos_checkpoint(self, rodada: int, rmse: float, is_best: bool):
        """Salva pesos em .pt (PyTorch)."""
        from src.task import criar_modelo, set_weights

        modelo = criar_modelo(self.model_name)
        set_weights(modelo, self._pesos_rodada_atual)
        
        checkpoint = {
            "rodada": rodada,
            "rmse": float(rmse),
            "model_state_dict": modelo.state_dict(),
            "timestamp": datetime.now().isoformat(),
        }
        
        if is_best:
            filename = "melhores_pesos.pt"
        else:
            filename = f"checkpoint_rodada_{rodada:03d}.pt"
        
        filepath = self.pesos_dir / filename
        torch.save(checkpoint, filepath)
        print(f"[ExperimentTracker] Pesos salvos: {filename}")
    
    def _salvar_metricas_json(self):
        """Atualiza JSON com historico completo."""
        resumo = {
            "experiment_name": self.experiment_name,
            "strategy": self.strategy_name,
            "model": self.model_name,
            "total_rodadas": len(self.historico),
            "melhor_rodada": self.melhor_rodada,
            "melhor_rmse": float(self.melhor_rmse),
            "historico": self.historico,
        }
        
        with open(self.metricas_file, "w", encoding="utf-8") as f:
            json.dump(resumo, f, indent=2, ensure_ascii=False)
    
    def exportar_metricas_csv(self):
        """Exporta historico em CSV para analise posterior."""
        if not self.historico:
            return
        
        with open(self.clientes_file, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["strategy", "model", "rodada", "rmse_global", "mse_global", "mae_global", "melhor", "timestamp"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for entrada in self.historico:
                writer.writerow({
                    "strategy": self.strategy_name,
                    "model": self.model_name,
                    "rodada": entrada["rodada"],
                    "rmse_global": entrada["rmse_global"],
                    "mse_global": entrada["mse_global"],
                    "mae_global": entrada.get("mae_global", float("nan")),
                    "melhor": entrada["melhor_ate_agora"],
                    "timestamp": entrada["timestamp"],
                })
        
        print(f"[ExperimentTracker] CSV salvo: {self.clientes_file.name}")
    
    def gerar_resumo_final(self):
        """Gera resumo executivo apos termino."""
        resumo = {
            "experiment": self.experiment_name,
            "strategy": self.strategy_name,
            "model": self.model_name,
            "total_rodadas": len(self.historico),
            "melhor_rodada": self.melhor_rodada,
            "melhor_rmse": float(self.melhor_rmse),
            "rmse_inicial": float(self.historico[0]["rmse_global"]) if self.historico else None,
            "melhoria_percentual": (
                (self.historico[0]["rmse_global"] - self.melhor_rmse) / 
                self.historico[0]["rmse_global"] * 100
            ) if self.historico and self.historico[0]["rmse_global"] > 0 else 0,
            "path": str(self.experiment_dir.resolve()),
        }
        
        print("\n" + "="*70)
        print(f"RESUMO DO EXPERIMENTO: {self.experiment_name}")
        print("="*70)
        print(f"Rodadas: {resumo['total_rodadas']}")
        print(f"Melhor RMSE: {resumo['melhor_rmse']:.4f}C (rodada {resumo['melhor_rodada']})")
        print(f"Melhoria: {resumo['melhoria_percentual']:.1f}%")
        print(f"Salvos em: {resumo['path']}")
        print("="*70 + "\n")
        
        return resumo
