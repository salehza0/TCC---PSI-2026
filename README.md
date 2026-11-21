# Projeto de Formatura: Sistema de Previsão de Temperatura Baseado em Aprendizagem Federada

Repositório dos códigos desenvolvidos para o Projeto de Formatura (PSI3592) do Departamento de Engenharia de Sistemas Eletrônicos da Escola Politécnica da USP.

**Autor:** Násser Assad Ferreira Saleh

## Sobre o projeto

O trabalho compara três paradigmas de treinamento (Local, Centralizado e Federado) na previsão de temperatura de curto prazo, usando dados horários de cinco estações do INMET no Estado de São Paulo. São avaliadas as arquiteturas LSTM e TCN e quatro estratégias de agregação federada (FedAvg, FedMedian, FedYogi e FedAdam), com o framework [Flower](https://flower.ai/) e PyTorch.

## Estrutura do repositório

| Pasta | Conteúdo |
|-------|----------|
| `Metodologia/` | Trabalho principal: implementação e experimentos dos três paradigmas. O README dessa pasta detalha a execução. |
| `PoC/` | Prova de conceito com dados sintéticos de duas estações, usada para validar a infraestrutura federada. |
| `Tratamento de dados/` | Scripts de seleção das estações e de pré-processamento dos dados brutos. |
| `Dados 2020-2025/` | Dados tratados das estações selecionadas, prontos para uso nos experimentos. |


## Fonte dos dados

Dados meteorológicos públicos do [INMET](https://portal.inmet.gov.br/).
