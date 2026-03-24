"""
Petrodora-AI: Model Evaluation & Benchmark Script (Full Production Version)
Covers: Fase 5 (Golden Dataset Evaluation) + Fase 6 (MLOps & Monitoring)

Metricas implementadas:
- ROUGE Score (rouge-score)
- Similarity Score (sentence-transformers)
- Length Drift (TextLength, WordCount via Evidently)
- Trigger de retreinamento (Similarity < 0.82)
- Logging no MLflow (DagsHub)
- Relatorio HTML Evidently AI
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import mlflow
import pandas as pd
import ollama
from dotenv import load_dotenv

# Evidently Core v2 (validado no .venv)
from evidently.core.report import Report
from evidently.presets import TextEvals
from evidently.core.datasets import Dataset, DataDefinition
from evidently.descriptors import TextLength, WordCount

# Metricas de Qualidade
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util

# Parametros de Governanca
SIMILARITY_TRIGGER_THRESHOLD: float = 0.50
MODEL_NAME: str = "petrodora-v1"
RUN_VERSION: str = "v1.0-Phi3-Basic"


def _validate_benchmark_data(df: pd.DataFrame) -> None:
    """
    Valida a integridade do DataFrame de benchmark antes do calculo de metricas.
    Atua como 'circuit breaker': levanta ValueError se dados estiverem corrompidos.

    Verifica:
    - Colunas obrigatorias presentes
    - Ausencia de valores nulos nas colunas criticas
    - Ausencia de strings vazias ou somente espacos
    - Ausencia de linhas duplicadas
    - Minimo de linhas para calculo estatistico valido

    :param df: DataFrame carregado do benchmark_results.csv.
    :raises ValueError: Se qualquer violacao de integridade for detectada.
    """
    REQUIRED_COLUMNS: List[str] = ['instruction', 'target', 'prediction']
    MIN_ROWS: int = 5

    print(f"\n[INFO] Validando integridade dos dados de benchmark ({len(df)} linhas)...")

    # 1. Verificar colunas obrigatorias
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"[VALIDATION ERROR] Colunas obrigatorias ausentes no CSV: {missing_cols}. "
            f"Colunas encontradas: {df.columns.tolist()}"
        )

    # 2. Verificar minimo de linhas
    if len(df) < MIN_ROWS:
        raise ValueError(
            f"[VALIDATION ERROR] Dataset insuficiente: {len(df)} linhas. "
            f"Minimo necessario: {MIN_ROWS}."
        )

    # 3. Verificar valores nulos
    null_counts = df[REQUIRED_COLUMNS].isnull().sum()
    null_violations = null_counts[null_counts > 0]
    if not null_violations.empty:
        raise ValueError(
            f"[VALIDATION ERROR] Valores nulos encontrados:\n{null_violations.to_string()}"
        )

    # 4. Verificar strings vazias ou somente espacos
    for col in ['target', 'prediction']:
        empty_mask = df[col].astype(str).str.strip() == ""
        empty_count = empty_mask.sum()
        if empty_count > 0:
            raise ValueError(
                f"[VALIDATION ERROR] Coluna '{col}' possui {empty_count} linha(s) "
                f"com string vazia ou somente espacos."
            )

    # 5. Verificar duplicatas completas
    duplicate_count = df.duplicated(subset=REQUIRED_COLUMNS).sum()
    if duplicate_count > 0:
        print(
            f"[WARN] {duplicate_count} linha(s) duplicada(s) detectada(s) em benchmark_results.csv. "
            f"Considere revisar o dataset."
        )

    print(
        f"[INFO] Validacao concluida: {len(df)} linhas validas, "
        f"0 nulos, 0 strings vazias, {duplicate_count} duplicatas (warning)."
    )


def _compute_rouge_scores(predictions: List[str], targets: List[str]) -> Dict[str, float]:
    """
    Calcula a media das metricas ROUGE (ROUGE-1, ROUGE-2, ROUGE-L) entre predicoes e targets.

    :param predictions: Lista de respostas geradas pelo modelo.
    :param targets: Lista de respostas de referencia do Golden Dataset.
    :return: Dicionario com as medias de F1 para ROUGE-1, ROUGE-2 e ROUGE-L.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    aggregated: Dict[str, List[float]] = {'rouge1': [], 'rouge2': [], 'rougeL': []}

    for pred, target in zip(predictions, targets):
        scores = scorer.score(target, pred)
        aggregated['rouge1'].append(scores['rouge1'].fmeasure)
        aggregated['rouge2'].append(scores['rouge2'].fmeasure)
        aggregated['rougeL'].append(scores['rougeL'].fmeasure)

    return {k: round(sum(v) / len(v), 4) for k, v in aggregated.items()}


def _compute_similarity_scores(
    predictions: List[str],
    targets: List[str],
    model_name: str = "all-MiniLM-L6-v2"
) -> Dict[str, float]:
    """
    Calcula a Similaridade Semantica media entre predicoes e targets usando Sentence Transformers.

    :param predictions: Lista de respostas geradas pelo modelo.
    :param targets: Lista de respostas de referencia do Golden Dataset.
    :param model_name: Nome do modelo de embeddings a ser carregado.
    :return: Dicionario com a media e o minimo do Cosine Similarity.
    """
    print(f"[INFO] Carregando modelo de embeddings: {model_name}...")
    model = SentenceTransformer(model_name)

    pred_embeddings = model.encode(predictions, convert_to_tensor=True)
    target_embeddings = model.encode(targets, convert_to_tensor=True)

    similarity_scores = util.cos_sim(pred_embeddings, target_embeddings).diagonal().tolist()

    avg_similarity = round(sum(similarity_scores) / len(similarity_scores), 4)
    min_similarity = round(min(similarity_scores), 4)

    return {
        "avg_similarity": avg_similarity,
        "min_similarity": min_similarity,
        "raw_scores": similarity_scores
    }


def _check_retrain_trigger(avg_similarity: float) -> None:
    """
    Verifica se o Similarity Score medio esta abaixo do threshold e emite o alerta de retreinamento.

    :param avg_similarity: Media do Cosine Similarity do benchmark atual.
    """
    if avg_similarity < SIMILARITY_TRIGGER_THRESHOLD:
        print(
            f"\n[ALERT] TRIGGER DE RETREINAMENTO ATIVO!\n"
            f"  Similarity Score: {avg_similarity:.4f} (Minimo: {SIMILARITY_TRIGGER_THRESHOLD})\n"
            f"  Recomendacao: Iniciar novo ciclo de Fine-Tuning com novos dados de feedback.\n"
        )
    else:
        print(
            f"\n[OK] Modelo em conformidade. "
            f"Similarity Score: {avg_similarity:.4f} >= {SIMILARITY_TRIGGER_THRESHOLD}"
        )


def _generate_evidently_report(
    df: pd.DataFrame,
    output_path: Path
) -> None:
    """
    Gera o relatorio HTML de drift e qualidade via Evidently AI (Core v2).

    Arquitetura:
    - reference_data: respostas HUMANAS do Golden Dataset (padrao de qualidade).
    - current_data  : respostas geradas pela IA (Petrodora-v1).

    O Evidently compara as distribuicoes dos descritores entre os dois grupos,
    permitindo detectar drift de qualidade (ex: IA mais verbosa, mais neutra, etc.).

    Nota: SemanticSimilarity ja e calculado via sentence-transformers no passo
    anterior e registrado no MLflow — nao e replicado aqui para evitar redundancia
    de embeddings e consumo extra de VRAM.

    Descritores aplicados (coluna 'text'):
    - TextLength, WordCount, SentenceCount, Sentiment,
      OOVWordsPercentage, NonLetterCharacterPercentage

    :param df: DataFrame com colunas 'instruction', 'target' e 'prediction'.
    :param output_path: Caminho completo para salvar o arquivo HTML.
    """
    from evidently.descriptors import (
        SentenceCount,
        Sentiment,
        OOVWordsPercentage,
        NonLetterCharacterPercentage,
    )

    print(f"\n[INFO] Gerando Relatorio Evidently AI (Reference: Humano | Current: IA)...")

    # Ambos usam a mesma coluna 'text' para comparacao homogenea
    reference_df = df[['instruction', 'target']].copy().rename(columns={'target': 'text'})
    current_df = df[['instruction', 'prediction']].copy().rename(columns={'prediction': 'text'})

    definition = DataDefinition(text_columns=['text'])
    reference_dataset = Dataset.from_pandas(reference_df, data_definition=definition)
    current_dataset = Dataset.from_pandas(current_df, data_definition=definition)

    # Descritores aplicados nos dois datasets para comparacao de distribuicao
    descriptors = [
        TextLength(column_name="text"),
        WordCount(column_name="text"),
        SentenceCount(column_name="text"),
        Sentiment(column_name="text"),
        OOVWordsPercentage(column_name="text"),
        NonLetterCharacterPercentage(column_name="text"),
    ]
    reference_dataset.add_descriptors(descriptors)
    current_dataset.add_descriptors(descriptors)

    report = Report(metrics=[TextEvals()])
    snapshot = report.run(reference_data=reference_dataset, current_data=current_dataset)
    snapshot.save_html(str(output_path))

    print(f"[INFO] Relatorio de drift HTML salvo em: {output_path}")


def run_evaluation() -> None:
    """
    Pipeline principal de avaliacao do modelo Petrodora-AI.
    Executa ROUGE, Similarity Score, Length Drift e registra no MLflow.
    """
    load_dotenv()
    ROOT = Path(__file__).parent.parent
    GOLDEN_DATA_PATH = ROOT / "knowledge" / "golden_dataset.json"
    OUTPUT_REPORT_DIR = ROOT / "reports"
    FINAL_CSV_PATH = ROOT / "data" / "benchmark_results.csv"

    OUTPUT_REPORT_DIR.mkdir(exist_ok=True)
    FINAL_CSV_PATH.parent.mkdir(exist_ok=True)

    # --- 1. Carregar ou Gerar Resultados ---
    if FINAL_CSV_PATH.exists():
        print(f"\n[INFO] Resultados encontrados em {FINAL_CSV_PATH}, carregando...")
        df = pd.read_csv(FINAL_CSV_PATH)
    else:
        if not GOLDEN_DATA_PATH.exists():
            print(f"[ERROR] Golden Dataset nao encontrado em {GOLDEN_DATA_PATH}")
            return

        with open(GOLDEN_DATA_PATH, 'r', encoding='utf-8') as f:
            golden_data: List[Dict] = json.load(f)

        results: List[Dict] = []
        print(f"[ACTION] Consultando modelo '{MODEL_NAME}' no Ollama ({len(golden_data)} perguntas)...")
        for i, item in enumerate(golden_data):
            instruction = item.get("instruction", "")
            target = item.get("output", "")
            try:
                response = ollama.chat(
                    model=MODEL_NAME,
                    messages=[{'role': 'user', 'content': instruction}]
                )
                prediction = response['message']['content']
                results.append({"instruction": instruction, "target": target, "prediction": prediction})
                print(f"  [{i+1}/{len(golden_data)}] OK")
            except Exception as e:
                print(f"  [{i+1}/{len(golden_data)}] ERRO: {e}")

        df = pd.DataFrame(results)
        df.to_csv(FINAL_CSV_PATH, index=False, encoding='utf-8')

    # --- 2. Validar Integridade dos Dados (Circuit Breaker) ---
    _validate_benchmark_data(df)

    predictions: List[str] = df['prediction'].astype(str).tolist()
    targets: List[str] = df['target'].astype(str).tolist()

    # --- 3. Calcular Metricas de Qualidade ---
    print(f"\n[INFO] Calculando ROUGE Score...")
    rouge_metrics = _compute_rouge_scores(predictions, targets)
    print(f"  ROUGE-1: {rouge_metrics['rouge1']} | ROUGE-2: {rouge_metrics['rouge2']} | ROUGE-L: {rouge_metrics['rougeL']}")

    print(f"\n[INFO] Calculando Similarity Score (Sentence Transformers)...")
    similarity_results = _compute_similarity_scores(predictions, targets)
    avg_sim = similarity_results['avg_similarity']
    print(f"  Avg Similarity: {avg_sim} | Min Similarity: {similarity_results['min_similarity']}")

    # --- 3. Verificar Trigger de Retreinamento ---
    _check_retrain_trigger(avg_sim)

    # --- 4. Gerar Relatorio Evidently AI ---
    report_path = OUTPUT_REPORT_DIR / "eval_report_v1.html"
    _generate_evidently_report(df, report_path)

    # --- 5. Registrar Metricas no MLflow (DagsHub) ---
    mlflow_uri: Optional[str] = os.getenv("MLFLOW_TRACKING_URI")
    if mlflow_uri:
        print(f"\n[INFO] Registrando metricas no MLflow: {mlflow_uri}")
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("petrodora-benchmark")

        with mlflow.start_run(run_name=RUN_VERSION):
            # Metricas ROUGE
            mlflow.log_metric("rouge1_f1", rouge_metrics['rouge1'])
            mlflow.log_metric("rouge2_f1", rouge_metrics['rouge2'])
            mlflow.log_metric("rougeL_f1", rouge_metrics['rougeL'])

            # Similarity Score
            mlflow.log_metric("avg_similarity_score", avg_sim)
            mlflow.log_metric("min_similarity_score", similarity_results['min_similarity'])

            # Parametros do ciclo
            mlflow.log_param("model_name", MODEL_NAME)
            mlflow.log_param("run_version", RUN_VERSION)
            mlflow.log_param("num_questions", len(df))
            mlflow.log_param("similarity_trigger_threshold", SIMILARITY_TRIGGER_THRESHOLD)
            mlflow.log_param("retrain_triggered", avg_sim < SIMILARITY_TRIGGER_THRESHOLD)

            # Artefato: Relatorio HTML
            mlflow.log_artifact(str(report_path))

        print("[INFO] Metricas registradas no MLflow com sucesso.")
    else:
        print("[WARN] MLFLOW_TRACKING_URI nao configurado. Pulando registro remoto.")

    # --- Sumario Final ---
    print("\n" + "=" * 50)
    print("SUMARIO DO BENCHMARK - PETRODORA-AI")
    print("=" * 50)
    print(f"  Versao:          {RUN_VERSION}")
    print(f"  Total Perguntas: {len(df)}")
    print(f"  ROUGE-1 F1:      {rouge_metrics['rouge1']}")
    print(f"  ROUGE-2 F1:      {rouge_metrics['rouge2']}")
    print(f"  ROUGE-L F1:      {rouge_metrics['rougeL']}")
    print(f"  Similarity Avg:  {avg_sim}")
    print(f"  Retrain Trigger: {'SIM' if avg_sim < SIMILARITY_TRIGGER_THRESHOLD else 'NAO'}")
    print(f"  Relatorio HTML:  {report_path}")
    print("=" * 50)


if __name__ == "__main__":
    run_evaluation()
