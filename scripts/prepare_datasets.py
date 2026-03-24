import json
import random
import logging
from pathlib import Path
from typing import List, Dict

# Configuração de Logging (Padrão Omenortep-AI)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def split_dataset(input_path: Path, training_path: Path, golden_path: Path, golden_size: int = 20):
    """
    Divide o dataset gerado em dados de treinamento e um benchmark (Golden Dataset).
    Garante que os dados de teste não estejam presentes no treino (Data Leakage Protection).
    """
    if not input_path.exists():
        logger.error(f"Dataset de entrada não encontrado: {input_path}")
        return

    # Lendo todas as entradas JSONL
    logger.info(f"Lendo dataset: {input_path.name}")
    entries: List[Dict[str, str]] = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    if len(entries) < golden_size:
        logger.warning(f"O dataset possui apenas {len(entries)} entradas. Golden de {golden_size} não é possível.")
        return

    # Embaralhando para garantir diversidade nos dois conjuntos
    random.seed(42) # Seed fixa para reprodutibilidade
    random.shuffle(entries)

    # Dividindo (20 para Golden, Restante para Treino)
    golden_dataset = entries[:golden_size]
    training_dataset = entries[golden_size:]

    # Salvando Golden Dataset como JSON (Array de objetos) para auditoria simplificada
    with open(golden_path, 'w', encoding='utf-8') as f:
        json.dump(golden_dataset, f, ensure_ascii=False, indent=4)
    logger.info(f"Golden Dataset (Benchmark) salvo: {golden_size} entradas em {golden_path}")

    # Salvando Training Data como JSONL (Formato padrão de Fine-tuning)
    with open(training_path, 'w', encoding='utf-8') as f:
        for item in training_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    logger.info(f"Training Dataset salvo: {len(training_dataset)} entradas em {training_path}")

if __name__ == "__main__":
    # Caminhos Petrodora-AI
    ROOT_DIR = Path(__file__).parent.parent
    INPUT = ROOT_DIR / "data" / "processed" / "training_data.jsonl"
    TRAIN_OUT = ROOT_DIR / "data" / "processed" / "training_data.jsonl"
    GOLDEN_OUT = ROOT_DIR / "knowledge" / "golden_dataset.json"

    split_dataset(INPUT, TRAIN_OUT, GOLDEN_OUT)
