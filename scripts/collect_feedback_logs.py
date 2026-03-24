"""
Petrodora-AI: Feedback Log Collector
Purpose: Extrai historico de conversas do Open WebUI (SQLite via Docker volume)
         e salva em /feedback/logs para analise e retreinamento.
         Ao atingir 50 interacoes acumuladas, dispara automaticamente
         o pipeline de avaliacao (evaluate_model.py).

Uso:
    python scripts/collect_feedback_logs.py

Dependencias externas:
    - Docker rodando com o volume petrodora-ai_open-webui ativo
    - Ollama rodando com petrodora-v1 (para o trigger de avaliacao)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Configuracao de Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
DOCKER_VOLUME_NAME: str = "petrodora-ai_open-webui"
DB_PATH_IN_VOLUME: str = "/data/webui.db"
INTERACTION_TRIGGER_THRESHOLD: int = 50
STATE_FILE_NAME: str = "last_sync_state.json"
LOG_FILE_PREFIX: str = "feedback"


# ---------------------------------------------------------------------------
# Funcoes de Suporte
# ---------------------------------------------------------------------------

def _get_project_root() -> Path:
    """Retorna a raiz do projeto (pasta pai de scripts/)."""
    return Path(__file__).parent.parent


def _get_state_file_path(feedback_dir: Path) -> Path:
    """Retorna o caminho do arquivo de estado de sincronizacao."""
    return feedback_dir / "state" / STATE_FILE_NAME


def _load_sync_state(state_path: Path) -> Dict[str, int]:
    """
    Carrega o estado da ultima sincronizacao.

    :param state_path: Caminho do arquivo JSON de estado.
    :return: Dicionario com 'last_sync_timestamp' e 'total_interactions'.
    """
    if state_path.exists():
        with open(state_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"last_sync_timestamp": 0, "total_interactions": 0}


def _save_sync_state(state_path: Path, state: Dict[str, int]) -> None:
    """
    Persiste o estado da sincronizacao atual.

    :param state_path: Caminho do arquivo JSON de estado.
    :param state: Dicionario com os dados de estado a salvar.
    """
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    logger.info(f"Estado de sincronizacao salvo em: {state_path}")


def _copy_db_from_volume(tmp_db_path: str) -> bool:
    """
    Copia o banco SQLite do volume Docker para um arquivo temporario local.
    Utiliza um container Alpine efemero para a transferencia.

    :param tmp_db_path: Caminho local de destino para o arquivo .db copiado.
    :return: True se a copia foi bem-sucedida, False caso contrario.
    """
    logger.info(f"Copiando banco de dados do volume '{DOCKER_VOLUME_NAME}'...")
    result = subprocess.run(
        [
            "docker", "run", "--rm",
            "-v", f"{DOCKER_VOLUME_NAME}:/data",
            "-v", f"{Path(tmp_db_path).parent}:/output",
            "alpine",
            "cp", DB_PATH_IN_VOLUME, f"/output/{Path(tmp_db_path).name}",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.error(f"Falha ao copiar o banco de dados: {result.stderr}")
        return False

    logger.info("Banco de dados copiado com sucesso.")
    return True


def _extract_interactions_from_db(
    db_path: str,
    since_timestamp: int,
) -> List[Dict[str, str]]:
    """
    Extrai pares de interacao (usuario/assistente) do banco SQLite do Open WebUI.

    Esquema da tabela 'chat':
        - id         : UUID da sessao
        - user_id    : ID do usuario
        - title      : Titulo gerado pelo modelo
        - chat       : JSON com lista de mensagens
        - created_at : Unix timestamp de criacao
        - updated_at : Unix timestamp de ultima atualizacao

    :param db_path: Caminho local do arquivo .db copiado.
    :param since_timestamp: Filtra apenas sessoes atualizadas apos este timestamp.
    :return: Lista de dicionarios com campos 'instruction', 'response', 'session_id', 'timestamp'.
    """
    interactions: List[Dict[str, str]] = []

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT id, title, chat, updated_at FROM chat WHERE updated_at > ? ORDER BY updated_at ASC",
            (since_timestamp,),
        )
        rows = cursor.fetchall()
        conn.close()

        logger.info(f"Sessoes novas encontradas desde ultimo sync: {len(rows)}")

        for session_id, title, chat_json, updated_at in rows:
            try:
                chat_data: Dict = json.loads(chat_json)
                messages: List[Dict] = chat_data.get("messages", [])

                # Extrair pares sequenciais user -> assistant
                for i in range(len(messages) - 1):
                    user_msg = messages[i]
                    asst_msg = messages[i + 1]

                    if user_msg.get("role") == "user" and asst_msg.get("role") == "assistant":
                        user_content = user_msg.get("content", "").strip()
                        asst_content = asst_msg.get("content", "").strip()

                        if user_content and asst_content:
                            interactions.append({
                                "session_id": session_id,
                                "title": title or "",
                                "instruction": user_content,
                                "response": asst_content,
                                "timestamp": updated_at,
                            })

            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Sessao {session_id}: erro ao parsear chat JSON: {e}")

    except sqlite3.Error as e:
        logger.error(f"Erro ao acessar o banco SQLite: {e}")

    return interactions


def _save_interactions_to_log(
    interactions: List[Dict[str, str]],
    feedback_dir: Path,
) -> Optional[Path]:
    """
    Salva as interacoes extraidas em um arquivo JSONL com timestamp no nome.

    :param interactions: Lista de pares de interacao.
    :param feedback_dir: Diretorio de destino para os logs.
    :return: Caminho do arquivo salvo, ou None se nao houver interacoes.
    """
    if not interactions:
        logger.info("Nenhuma interacao nova para salvar.")
        return None

    feedback_dir.mkdir(parents=True, exist_ok=True)
    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_file = feedback_dir / f"{LOG_FILE_PREFIX}_{timestamp_str}.jsonl"

    with open(log_file, "w", encoding="utf-8") as f:
        for entry in interactions:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(f"Log salvo: {log_file} ({len(interactions)} interacoes)")
    return log_file


def _count_total_logged_interactions(feedback_dir: Path) -> int:
    """
    Conta o total de interacoes registradas em todos os arquivos JSONL de feedback.

    :param feedback_dir: Diretorio contendo os arquivos de log.
    :return: Total de linhas (interacoes) em todos os arquivos .jsonl.
    """
    total: int = 0
    for log_file in feedback_dir.glob(f"{LOG_FILE_PREFIX}_*.jsonl"):
        with open(log_file, "r", encoding="utf-8") as f:
            total += sum(1 for line in f if line.strip())
    return total


def _trigger_evaluation_pipeline(scripts_dir: Path) -> None:
    """
    Dispara o pipeline de avaliacao (evaluate_model.py) quando o
    threshold de interacoes e atingido.

    :param scripts_dir: Diretorio contendo o script de avaliacao.
    """
    eval_script = scripts_dir / "evaluate_model.py"
    if not eval_script.exists():
        logger.error(f"Script de avaliacao nao encontrado: {eval_script}")
        return

    logger.info(
        f"\n{'=' * 50}\n"
        f"TRIGGER ATIVO: {INTERACTION_TRIGGER_THRESHOLD} interacoes atingidas.\n"
        f"Iniciando pipeline de avaliacao automatica...\n"
        f"{'=' * 50}"
    )
    result = subprocess.run(
        [sys.executable, str(eval_script)],
        capture_output=False,  # Permite saida direta no terminal
    )
    if result.returncode == 0:
        logger.info("Pipeline de avaliacao concluido com sucesso.")
    else:
        logger.error(f"Pipeline de avaliacao encerrado com erro (code: {result.returncode}).")


# ---------------------------------------------------------------------------
# Pipeline Principal
# ---------------------------------------------------------------------------

def collect_feedback_logs() -> None:
    """
    Pipeline principal de coleta de logs do Open WebUI.

    Fluxo:
        1. Carrega estado da ultima sincronizacao.
        2. Copia o banco SQLite do volume Docker para temp.
        3. Extrai novas interacoes desde o ultimo timestamp.
        4. Salva em /feedback/logs/feedback_TIMESTAMP.jsonl.
        5. Atualiza o estado de sincronizacao.
        6. Conta total de interacoes acumuladas.
        7. Se total >= 50, dispara evaluate_model.py.
    """
    root = _get_project_root()
    feedback_dir = root / "feedback" / "logs"
    state_path = _get_state_file_path(root / "feedback")
    scripts_dir = root / "scripts"

    # 1. Carregar estado anterior
    state = _load_sync_state(state_path)
    last_sync = state["last_sync_timestamp"]
    logger.info(f"Ultimo sync: {datetime.fromtimestamp(last_sync, tz=timezone.utc).isoformat() if last_sync else 'nunca'}")

    # 2. Copiar banco do volume Docker para arquivo temporario
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_db = str(Path(tmp_dir) / "webui.db")

        if not _copy_db_from_volume(tmp_db):
            logger.error("Abortando: nao foi possivel acessar o banco do Open WebUI.")
            logger.error("Verifique se o Docker esta rodando e o volume 'petrodora-ai_open-webui' existe.")
            return

        # 3. Extrair interacoes novas
        interactions = _extract_interactions_from_db(tmp_db, since_timestamp=last_sync)

    if not interactions:
        logger.info("Nenhuma interacao nova desde o ultimo sync. Encerrando.")
        return

    # 4. Salvar log JSONL
    _save_interactions_to_log(interactions, feedback_dir)

    # 5. Atualizar estado
    max_timestamp = max(entry["timestamp"] for entry in interactions)
    state["last_sync_timestamp"] = max_timestamp
    state["total_interactions"] = state.get("total_interactions", 0) + len(interactions)
    _save_sync_state(state_path, state)

    # 6. Contar total acumulado nos arquivos de log
    total_logged = _count_total_logged_interactions(feedback_dir)
    logger.info(f"Total de interacoes acumuladas nos logs: {total_logged}")

    # 7. Verificar trigger de avaliacao
    if total_logged >= INTERACTION_TRIGGER_THRESHOLD:
        _trigger_evaluation_pipeline(scripts_dir)
    else:
        remaining = INTERACTION_TRIGGER_THRESHOLD - total_logged
        logger.info(
            f"Avaliacao automatica em: {remaining} interacoes restantes "
            f"(threshold: {INTERACTION_TRIGGER_THRESHOLD})."
        )


if __name__ == "__main__":
    collect_feedback_logs()
