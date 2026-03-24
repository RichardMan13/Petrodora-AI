"""
Petrodora-AI: Phi-3 Fine-Tuning Script with Unsloth & MLflow
Purpose: Fine-tune Phi-3 Mini for O&G specialization using the Refined Training Data.
Author: Antigravity Assistant
"""

import os
import torch
import json
import mlflow
import dagshub
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from dotenv import load_dotenv
from pathlib import Path

# Carrega variáveis (.env) para DagsHub/MLflow
load_dotenv()

def train_petrodora():
    # 1. Configurar Tracking via DagsHub (MLOps)
    # Garante que o experimento apareça no DagsHub
    dagshub.init(repo_owner="RichardMan13", repo_name="Petrodora-AI", mlflow=True)
    
    # 2. Configurações do Modelo
    model_name = "unsloth/phi-3-mini-4k-instruct-bnb-4bit"
    max_seq_length = 2048
    dtype = None # None: auto-detect (Float16 ou Bfloat16)
    load_in_4bit = True

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    # 3. Adicionar Adaptadores LoRA
    # r=16: Equilíbrio entre aprendizado de novos fatos e economia de memória
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth", 
        random_state = 3407,
    )

    # 4. Carregar e Formatar Dataset Alpaca (No-RAG Strategy)
    # A estrutura instruct -> output sem o campo input para forçar conhecimento paramétrico
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        outputs      = examples["output"]
        texts = []
        for instruction, output in zip(instructions, outputs):
            # Template optimizado para Phi-3 style instruction tuning
            text = f"<|user|>\n{instruction}<|end|>\n<|assistant|>\n{output}<|end|>"
            texts.append(text)
        return { "text" : texts, }

    # Ajuste o caminho se necessário (para o Colab coloque na pasta raiz)
    data_path = "data/processed/training_data.jsonl"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Arquivo {data_path} não encontrado no diretório atual.")

    dataset = load_dataset("json", data_files=data_path, split="train")
    dataset = dataset.map(formatting_prompts_func, batched = True,)

    # 5. Configurar Treinamento (SFTTrainer)
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False, # Evita misturar diferentes exemplos na mesma janela (útil para datasets curtos)
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 10,
            max_steps = 60, # ~15 minutos no Colab / 20 minutos no local com GPU compatível
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            report_to = "none", # Manual logging no MLflow para evitar bugs de integração
        ),
    )

    # 6. Ciclo de Treino com MLflow Tracking
    with mlflow.start_run(run_name="v1.0-Initial-Tune-Phi3"):
        # Logar hiperparâmetros básicos
        mlflow.log_param("max_steps", 60)
        mlflow.log_param("lora_r", 16)
        
        print("\n[INFO] Iniciando Fine-Tuning do Petrodora-AI...")
        trainer.train()
        
        # 7. Salvar e Exportar (Formato Unsloth)
        model_save_path = "models/v1.0-Phi3-Basic"
        print(f"\n[SUCCESS] Treino concluído! Salvando modelo em {model_save_path}...")
        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        
        # Logar modelo final (opcional, como artefato)
        # mlflow.log_artifact(model_save_path)

if __name__ == "__main__":
    train_petrodora()
