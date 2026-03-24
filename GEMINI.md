# Petrodora-AI: Gemini Context & Guidelines

Este arquivo serve como o **Guia de Contexto Mestre** para a interação de IA (Gemini/Antigravity) neste projeto. Ele define quem somos, o que estamos construindo e as restrições técnicas intransponíveis.

---

## Contexto do Projeto
O **Petrodora-AI** é um projeto de especialização de **Small Language Models (SLMs)** para o domínio técnico de **Engenharia de Petróleo e Gás**. 

**Missão:** Criar um assistente técnico de alta precisão que rode localmente em hardware doméstico limitado, garantindo 100% de privacidade para manuais e normas técnicas proprietárias.

---

## Stack Tecnológica (Imutável)
Ao sugerir código ou arquitetura, **sempre** utilize estas bibliotecas:

- **Modelo Base:** `unsloth/phi-3-mini-4k-instruct-bnb-4bit`.
- **Treino:** **Unsloth** (QLoRA).
- **Inferência:** **Ollama** (GGUF quantização 4-bit).
- **Interface:** **Open WebUI** (Docker Compose).
- **OCR:** **Tesseract** ou **Docling** (processamento de PDFs antigos).
- **Métricas:** **ROUGE**, **BERTScore**, **Similarity Score** e **Length Drift**.
- **MLOps:** **MLflow** (DagsHub) para Tracking e **Evidently AI** para Monitoramento.
- **Hardware Target:** **NVIDIA GTX 1650 (4GB VRAM)**, Windows 10/11.

---

## Diretrizes de Arquitetura

1.  **Prioridade VRAM:** Todo script Python ou configuração do Docker deve ser otimizado para não exceder 4GB de VRAM. Prefira quantizações pesadas e limpeza agressiva de KV Cache.
2.  **No-RAG Strategy:** O projeto foca em **Parametric Knowledge** (conhecimento "congelado" nos pesos via Fine-Tuning). Não sugira arquiteturas de RAG ou bancos de vetores a menos que explicitamente solicitado.
3.  **Fluxo de Extração (ETL):** PDFs Brutos (`/data/raw`) ➡️ OCR/Parser ➡️ JSONL-Alpaca (`/data/processed`) ➡️ Validação (Golden Dataset).
4.  **Governança:** Nenhum ciclo de treino pode ser iniciado sem um `mlflow.start_run()` configurado para o servidor remoto.

---

## 📜 Regras de Comportamento para IA

- **Idioma:** Explicações em **Português (BR)**. Variáveis e termos técnicos no código em **Inglês**.
- **Estilo de Código:** Python com **Type Hinting** obrigatório e documentação técnica.
- **Restrição Estética:** **Não utilize emojis** dentro de strings de log, comentários de código ou variáveis.
- **Deteção de Drift:** Sempre pergunte sobre o **Golden Dataset** antes de validar um novo modelo.
- **Naming Convention:** Versione os treinamentos como `vX.X-Phi3-Contexto`.

---

## 🔗 Knowledge Items Prioritários
- Consulte sempre `.gemini/rules/` antes de gerar novos scripts.
- Verifique a pasta `/knowledge` para termos técnicos autorizados antes de gerar dados sintéticos para o dataset.
