# Petrodora-AI

Este repositório contém o plano de execução técnica e os recursos para o projeto de **Fine-Tuning e Eficiência On-Premises**, focado na especialização de modelos de linguagem para o domínio técnico de Engenharia de Petróleo e Gás.

## Objetivo

Desenvolvimento de um **SLM (Small Language Model)** especializado em jargões técnicos da indústria de petróleo e gás, operando localmente no **Windows** com hardware otimizado. O projeto garante privacidade total para manuais e normas técnicas proprietárias, sem dependência de bases de dados externas (estratégia No-RAG / Parametric Knowledge).

## Tecnologias Utilizadas

### IA e Fine-Tuning
- **Modelo Base:** `unsloth/phi-3-mini-4k-instruct-bnb-4bit` (Phi-3-mini).
- **Abstração:** **Unsloth** (Aceleração e eficiência de VRAM).
- **Técnica:** **QLoRA** (Fine-tuning de baixa precisão).
- **Plataforma de Treino:** **Google Colab** (T4/L4/A100 GPU).

### Deploy e Interface
- **Inferência:** **Ollama** (Servidor de LLM local, nativo no Windows).
- **Interface:** **Open WebUI** (Docker Compose).
- **Formato:** **GGUF** (Quantização Q4_K_M — otimizada para 4GB VRAM).
- **Orquestração:** **Docker Compose** no Windows.

### Extração e MLOps
- **OCR/Parser:** **Docling** (primário) + **pypdf** (fallback para PDFs corrompidos) + **Tesseract** ([Binários Windows](https://github.com/UB-Mannheim/tesseract/wiki)).
- **Linguagem:** **Python 3.10+** com Type Hinting obrigatório.
- **Tracking:** **MLflow** via **DagsHub** (Experiment Tracking).
- **Monitoramento:** **Evidently AI** (Drift de Qualidade e Descritores de Texto).

### Hardware Target
- **GPU:** NVIDIA GeForce GTX 1650 (4 GB VRAM).
- **SO:** Windows 10/11.

---

## Estrutura de Pastas

```
Petrodora-AI/
├── data/
│   ├── raw/                            # PDFs brutos e manuais técnicos originais
│   ├── processed/
│   │   ├── processed_knowledge.jsonl   # Saída do OCR/Parser (texto puro estruturado)
│   │   ├── training_data.jsonl         # Dataset de treino (80% - formato Alpaca)
│   │   └── golden_dataset.jsonl        # Dataset de benchmark (20%)
│   └── benchmark_results.csv           # Resultados do ciclo de avaliação atual
├── knowledge/
│   └── golden_dataset.json             # 20 pares Q&A revisados por especialistas
├── models/
│   └── *.gguf                          # Arquivo GGUF quantizado (Q4_K_M)
├── reports/
│   └── eval_report_v1.html             # Relatório HTML gerado pelo Evidently AI
├── feedback/
│   └── logs/                           # Logs de feedback e correções de usuários
├── scripts/
│   ├── ocr_extract.py                  # ETL: Extração de texto de PDFs (Docling + pypdf fallback)
│   ├── prepare_datasets.py             # ETL: Preparação e limpeza do dataset
│   ├── synth_generator.py              # Data Augmentation via GPT-4o/Claude
│   ├── split_dataset.py                # Divisão treino/benchmark (80/20)
│   ├── train_phi3_petrodora.py         # Fine-Tuning local (alternativa ao Colab)
│   ├── export_gguf.py                  # Exportação e quantização para formato GGUF
│   ├── evaluate_model.py               # Avaliação: ROUGE, Similarity, Evidently + MLflow
│   ├── test_mlflow_connection.py       # Utilitário: Validação da conexão com DagsHub
│   └── Petrodora_AI.ipynb              # Notebook oficial para Google Colab
├── .env.example                        # Template de variáveis de ambiente
├── docker-compose.yml                  # Configuração do Open WebUI
├── GEMINI.md                           # Guia de contexto master para a IA
└── requirements.txt
```

---

## Instalação e Requisitos

### Pré-requisitos
- **NVIDIA GPU:** GTX 1650 (4GB VRAM) com drivers atualizados.
- **Docker Desktop:** Instalado no Windows.
- **Python 3.10+**: Com `venv` configurado.
- **Ollama**: [Baixar Executável Windows](https://ollama.com/download/windows).

### Setup Rápido (Local)

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/RichardMan13/Petrodora-AI.git
   cd Petrodora-AI
   ```

2. **Variáveis de Ambiente:** Copie o `.env.example` para `.env` e preencha suas credenciais do **DagsHub**:
   ```bash
   # MLflow / DagsHub Credentials
   MLFLOW_TRACKING_URI=https://dagshub.com/RichardMan13/Petrodora-AI.mlflow
   MLFLOW_TRACKING_USERNAME=RichardMan13
   MLFLOW_TRACKING_PASSWORD=seu_token_aqui
   ```

3. **Ambiente Virtual:**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

4. **Interface UI (Open WebUI):**
   ```bash
   docker compose up -d
   ```

5. **Acesse:** [http://localhost:3000](http://localhost:3000)

---

## Plano de Execução Técnica

### Fase 1: Curadoria e Preparação do Dataset

O sucesso de um modelo especializado depende da qualidade dos dados e da abrangência da extração.

- **Fontes de Dados:** Manuais, normas e publicações técnicas via [Hub de Conhecimento IBP](https://www.ibp.org.br/hub-de-conhecimento/publicacoes/).
- **Pipeline de Extração (ETL):**
  - PDFs Brutos (`/data/raw`) → OCR/Parser (**Docling** primário, **pypdf** fallback) → JSONL-Alpaca (`/data/processed`).
  - Script: `scripts/ocr_extract.py`.
- **Estruturação (JSONL):** Dataset no padrão Alpaca (`instruction`, `input`, `output`) para instruções técnicas.
- **Data Augmentation:** Modelos robustos (GPT-4o/Claude) geram variações de perguntas via `scripts/synth_generator.py`.
- **Divisão Treino/Benchmark:** Script `scripts/split_dataset.py` (razão 80/20).

### Fase 2: Fine-Tuning com Unsloth (Google Colab)

- **Hardware de Treino:** T4 GPU (Google Colab) — sem consumo de VRAM local.
- **Modelo Base:** `unsloth/phi-3-mini-4k-instruct-bnb-4bit`.
- **Hiperparâmetros de referência:**
  - `max_seq_length`: 2048
  - `lora_alpha`: 16
  - `learning_rate`: 2e-4
- **Notebook:** `scripts/Petrodora_AI.ipynb` (upload para o Google Colab).

### Fase 3: Exportação e Quantização GGUF

- **Fusão de Pesos:** Geração direta do formato GGUF via Unsloth (`scripts/export_gguf.py`).
- **Quantização (Q4_K_M):** Otimizada para rodar em GPUs com 4GB de VRAM.
- **Persistência:** O arquivo `.gguf` é salvo em `models/` e registrado como artefato no MLflow.

### Fase 4: Deploy Local no Windows

- **Hardware de Destino:** NVIDIA GeForce GTX 1650 (4 GB VRAM).
- **Procedimento:**
  1. Garantir que o arquivo `.gguf` esteja em `models/`.
  2. O `Modelfile` já está configurado em `models/Modelfile` com o System Prompt do Engenheiro Sênior.
  3. Registrar o modelo: `ollama create petrodora-v1 -f models/Modelfile`.
  4. Rodar via CLI: `ollama run petrodora-v1`.
- **Interface (Open WebUI):** `docker compose up -d` — acesse em [http://localhost:3000](http://localhost:3000) e selecione `petrodora-v1` no seletor de modelos.
- **Notas do `docker-compose.yml`:**
  - `env_file` removido — o Docker Compose interpola `${VAR}` do `.env` automaticamente no bloco `environment`, sem expor credenciais desnecessárias ao container.
  - `healthcheck` com `start_period: 120s` configurado — evita falso status `unhealthy` durante o primeiro boot (migrações de DB + download de embeddings demoram 2-5 min).

### Fase 5: Avaliação e Refinamento

Validação rigorosa para garantir a utilidade real do modelo contra o Golden Dataset.

- **Golden Dataset:** `knowledge/golden_dataset.json` — 20 pares Q&A revisados por especialistas.
- **Métricas implementadas em `scripts/evaluate_model.py`:**

  | Métrica | Descrição |
  | :--- | :--- |
  | **ROUGE-1/2/L F1** | Sobreposição de n-gramas entre predição e referência |
  | **Similarity Score** | Cosine Similarity via `sentence-transformers` (all-MiniLM-L6-v2) |
  | **Length Drift** | Diferença de comprimento textual (TextLength, WordCount) |
  | **Sentiment Drift** | Desvio de polaridade entre resposta humana e da IA |
  | **OOV Percentage** | Percentual de palavras fora do vocabulário |

- **Circuit Breaker:** Validação de integridade dos dados (`_validate_benchmark_data`) antes do cálculo — bloqueia execução em caso de nulos, duplicatas ou strings vazias.
- **Trigger de Retreinamento:** Parametrizado em `SIMILARITY_TRIGGER_THRESHOLD = 0.50`. Alerta ativo se `avg_similarity < 0.50`.

### Fase 6: Governança e MLOps (MLflow & Evidently)

- **MLflow (Experiment Tracking):**
  - Infraestrutura: **DagsHub** para tracking remoto.
  - Nomenclatura obrigatória: `vX.X-Phi3-Contexto` (ex: `v1.0-Phi3-Basic`).
  - Logs: hiperparâmetros, métricas de qualidade e artefato HTML do Evidently.
  - Regra: nenhum ciclo de treino pode ser iniciado sem `mlflow.start_run()`.
- **Evidently AI (Monitoramento de Qualidade):**
  - Arquitetura: `reference_data` = respostas humanas (Golden Dataset) vs. `current_data` = respostas da IA.
  - Descritores: TextLength, WordCount, SentenceCount, Sentiment, OOVWordsPercentage, NonLetterCharacterPercentage.
  - Relatório HTML salvo em `reports/eval_report_v1.html` e logado no MLflow como artefato.
  - Frequência alvo: a cada 50 novas interações de teste ou uso real.

---

## Fluxo de Treinamento (Google Colab)

Para realizar novos ciclos de Fine-Tuning, utilize o notebook oficial:

- **Notebook:** `scripts/Petrodora_AI.ipynb` (Upload para o Google Colab).

Este notebook já contém as células de configuração para **GPU T4/L4/A100**, instalação do **Unsloth** e integração com **DagsHub/MLflow**.

---

## Comparativo Estratégico

| Métrica | Local (Petrodora-AI) | API Paga (GPT-4o) |
| :--- | :--- | :--- |
| **Hardware** | GTX 1650 (4GB) / Windows | Nuvem Robusta |
| **Privacidade** | 100% Local | Dados na nuvem do provedor |
| **Governança** | MLflow (DagsHub) + Evidently | Proprietária/Nula |
| **Monitoramento** | Similarity, ROUGE, Drift | Nula (Feedback Manual) |
| **Latência** | ~20-30 t/s (Estimado) | Variável |
| **Custo (1M tokens)** | $0.00 (Energia Local) | ~$5.00 - $15.00 |

---

## Padrões de Desenvolvimento (Governança)

Como definido no `GEMINI.md`, todo desenvolvimento deve seguir:

1. **Type Hinting:** Todo código Python deve ter tipagem estrita (`from __future__ import annotations`, `from typing import ...`).
2. **Linguagem:**
   - **Explicações/Comentários:** Português (BR).
   - **Código/Variáveis:** Inglês Técnico.
3. **Estética:** Proibição de emojis em variáveis, logs técnicos ou comentários de código.
4. **Tracking:** Todo ciclo de treino deve usar `mlflow.start_run()` configurado para o servidor remoto do DagsHub.
5. **Naming Convention:** Versionar treinamentos como `vX.X-Phi3-Contexto` (ex: `v1.0-Phi3-Basic`, `v2.0-Phi3-Expanded`).
6. **Circuit Breaker:** Toda pipeline de avaliação deve validar a integridade dos dados antes do cálculo de métricas.

---

## Checklist de Execução

### 1. Preparação (Fase 1)
- [x] Criar estrutura de pastas (`/data`, `/scripts`, `/models`, `/knowledge`, `/feedback`, `/reports`).
- [x] Mapear diretórios de manuais técnicos (PDFs) para `/data/raw`.
- [x] Implementar `scripts/ocr_extract.py` com Docling (primário) e pypdf (fallback).
- [x] Estruturar dataset inicial em `/data/processed/processed_knowledge.jsonl`.
- [x] Data Augmentation via `scripts/synth_generator.py` (100 pares via GPT-4o).
- [x] Separar dataset: treino (80) em `training_data.jsonl` e benchmark (20) em `golden_dataset.jsonl` via `scripts/split_dataset.py`.

### 2. Fine-Tuning & Tracking (Fase 2, 3 & 6)
- [x] Configurar conta no **DagsHub** e obter as credenciais do MLflow.
- [x] Configurar ambiente no Google Colab com Unsloth e MLflow.
- [x] Realizar treinamento com nomenclatura `v1.0-Phi3-Basic`.
- [x] Exportar arquivo `.gguf` (Q4_K_M) via `scripts/export_gguf.py`.
- [x] Validar conexão com DagsHub via `scripts/test_mlflow_connection.py`.

### 3. Validação & Monitoramento (Fase 5 & 6)
- [x] Criar Golden Dataset em `knowledge/golden_dataset.json` (20 pares técnicos).
- [x] Implementar **Circuit Breaker** de validação de dados (`_validate_benchmark_data`).
- [x] Calcular **ROUGE Score** (R1, R2, RL) em `scripts/evaluate_model.py`.
- [x] Calcular **Similarity Score** via `sentence-transformers` (all-MiniLM-L6-v2).
- [x] Gerar relatório HTML do **Evidently AI** (`reports/eval_report_v1.html`).
- [x] Logar todas as métricas no **MLflow/DagsHub** (experimento `petrodora-benchmark`).
- [x] Definir **trigger de retreinamento** (`SIMILARITY_TRIGGER_THRESHOLD = 0.50`).

### 4. Deploy (Fase 4)
- [x] Instalar Ollama no Windows (Nativo).
- [x] Registrar e rodar o modelo via `ollama create petrodora-v1 -f models/Modelfile`.
- [x] Criar `Modelfile` com System Prompt do Engenheiro Sênior em `models/`.
- [x] Corrigir `docker-compose.yml`: remover `env_file` e adicionar `healthcheck` com `start_period: 120s`.
- [x] Validar Open WebUI em [http://localhost:3000](http://localhost:3000) com `petrodora-v1:latest` respondendo em Português (BR).

### 5. Operação
- [ ] Integrar a coleta de logs da interface Open WebUI à pasta `/feedback/logs`.
- [x] Monitorar uso de VRAM na GTX 1650 — leitura em inferência: **3875 MB / 4096 MB (94.6%)** (em idle o Ollama descarrega o modelo automaticamente).
- [ ] Automatizar execução do relatório HTML a cada 50 novas interações.
- [ ] Expandir **Golden Dataset** para 50+ pares para cobertura de v2.0.
- [ ] Iniciar ciclo `v2.0-Phi3-Expanded` após coleta de feedback suficiente.

---