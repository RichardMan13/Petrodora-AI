import os
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv

# Carrega variáveis de ambiente (.env)
load_dotenv()

# Configuração de Logging (Padrão Petrodora-AI)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PetrodoraSynthGenerator:
    """
    Gerador de dados sintéticos para especialização de LLMs no domínio de O&G.
    Utiliza OpenAI (GPT-4o) para transformar texto extraído em pares de instrução/resposta.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Inicializa o gerador configurando a API da OpenAI.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY não encontrada no ambiente ou .env")
        
        self.client = OpenAI(api_key=self.api_key)
        # Modelo GPT-4o conforme o roadmap da Fase 1
        self.model = "gpt-4o"
        
        # Prompt de Sistema (Especialização em Engenharia de Petróleo)
        self.prompt_template = """
        Você é um Engenheiro de Petróleo e Gás Sênior.
        Seu objetivo é ler o fragmento de documentação técnica e gerar exatos 25 pares de Instrução e Resposta no estilo Alpaca.

        CONTEÚDO TÉCNICO:
        {content}

        DIRETRIZES:
        1. A 'instruction' deve ser um comando técnico ou pergunta profissional que possa ser respondida sem o texto original presente.
        2. O campo 'input' deve ser SEMPRE uma string vazia "".
        3. A 'output' deve ser uma resposta técnica, exata e rica em jargões do setor.
        4. Formato de Saída (JSON VÁLIDO):
        {{
            "instructions": [
                {{"instruction": "...", "input": "", "output": "..."}},
                ...
            ]
        }}
        
        Responda apenas com o JSON puro (objeto com a chave "instructions").
        """

    def generate_batch(self, content: str, source: str) -> List[Dict[str, str]]:
        """
        Envia um bloco de texto para o GPT e retorna a lista de instruções geradas.
        """
        try:
            prompt = self.prompt_template.format(content=content[:10000], source=source)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={ "type": "json_object" } # Garante saída JSON se o modelo suportar
            )
            
            raw_text = response.choices[0].message.content.strip()
            data = json.loads(raw_text)
            
            # Extração da chave 'instructions' conforme solicitado no prompt
            if isinstance(data, dict) and "instructions" in data:
                return data["instructions"]
            
            # Fallback caso ele retorne outra chave de lista
            if isinstance(data, dict):
                for key in data:
                    if isinstance(data[key], list):
                        return data[key]
            
            logger.warning(f"Resposta JSON vinda de {source} não continha uma lista válida.")
            return []
            
        except Exception as e:
            logger.error(f"Falha na geração sintética para {source}: {str(e)}")
            return []

    def run(self, input_path: Path, output_path: Path) -> None:
        """
        Lê o arquivo bruto do Docling e gera o dataset de treino final.
        """
        if not input_path.exists():
            logger.error(f"Arquivo de entrada não encontrado: {input_path}")
            return

        logger.info(f"Iniciando Geração Sintética (OpenAI): {input_path.name}")
        
        all_entries: List[Dict[str, str]] = []
        
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines):
            try:
                raw_data = json.loads(line)
                content = raw_data.get("input", "")
                
                # Extração do nome da fonte (arquivo original)
                source = "Manual Técnico"
                if "Documento de Origem:" in content:
                    source = content.split('\n')[0].replace("Documento de Origem: ", "").strip()

                logger.info(f"Processando entrada {i+1}/{len(lines)} (Fonte: {source})")
                
                synth_batch = self.generate_batch(content, source)
                all_entries.extend(synth_batch)
                
                # Pausa leve para controle de rate limits (opcional para GPT-4o)
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Erro na linha {i+1}: {str(e)}")

        # Salvando o resultado final em JSONL
        if all_entries:
            with open(output_path, 'w', encoding='utf-8') as f:
                for entry in all_entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            logger.info(f"Dataset de treinamento salvo com sucesso em: {output_path}")
            logger.info(f"Total de pares gerados: {len(all_entries)}")
        else:
            logger.warning("Nenhum dado sintético foi gerado.")

if __name__ == "__main__":
    # Caminhos Petrodora-AI
    ROOT_DIR = Path(__file__).parent.parent
    IN_FILE = ROOT_DIR / "data" / "processed" / "processed_knowledge.jsonl"
    OUT_FILE = ROOT_DIR / "data" / "processed" / "training_data.jsonl"

    try:
        generator = PetrodoraSynthGenerator()
        generator.run(IN_FILE, OUT_FILE)
    except Exception as e:
        logger.error(f"Erro fatal: {str(e)}")
