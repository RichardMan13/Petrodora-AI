"""
Petrodora-AI: GGUF Export & Quantization Script
Purpose: Merge LoRA adapters and quantization to Q4_K_M GGUF format for local execution (Ollama).
Author: Antigravity Assistant
"""

import os
from pathlib import Path
from unsloth import FastLanguageModel

def export_gguf():
    # Caminho do modelo treinado (Fase 2)
    ROOT = Path(__file__).parent.parent
    MODEL_PATH = ROOT / "models" / "v1.0-Phi3-Basic"
    OUTPUT_DIR = ROOT / "models" / "GGUF-Export"

    if not MODEL_PATH.exists():
        print(f"Erro: O modelo treinado não foi encontrado em {MODEL_PATH}.")
        print("Certifique-se de que o script train_phi3_petrodora.py terminou com sucesso.")
        return

    print(f"\n[INFO] Carregando modelo e adaptadores de: {MODEL_PATH}")
    
    # Carregar o modelo do diretório local no formato Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = str(MODEL_PATH),
        max_seq_length = 2048,
        load_in_4bit = True,
    )

    # 1. Exportar para GGUF (Quantização recomendada: q4_k_m para 4GB VRAM target)
    # A unsloth funde os pesos (Merge) e quantiza em um único passo.
    print(f"\n[ACTION] Realizando Fusão de Pesos e Quantização (Q4_K_M)...")
    
    try:
        model.save_pretrained_gguf(
            str(OUTPUT_DIR), 
            tokenizer, 
            quantization_method = "q4_k_m"
        )
        print(f"\n[SUCCESS] Modelo exportado com sucesso para: {OUTPUT_DIR}")
        print("-" * 50)
        print("PRÓXIMOS PASSOS:")
        print("1. Baixe o arquivo .gguf da pasta 'models/GGUF-Export/' no Colab para o seu PC.")
        print("2. Use o Ollama no Windows para importar o modelo (Fase 4 do README).")
        print("-" * 50)
    except Exception as e:
        print(f"\n[ERROR] Falha na exportação GGUF: {str(e)}")

if __name__ == "__main__":
    export_gguf()
