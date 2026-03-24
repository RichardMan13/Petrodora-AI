import os
import json
import logging
import torch
import gc
from pathlib import Path
from typing import List, Dict, Any, Optional
from pypdf import PdfReader
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

# Desativa alertas de symlinks no Windows para evitar poluição visual nos logs
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Configuração de Logging (Padrão de Governança Petrodora-AI)
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PetrodoraExtractor:
    """
    Extrator de alta precisão para documentos de Engenharia de Petróleo.
    Utiliza Docling com aceleração GPU para manter hierarquia de tabelas e layouts complexos.
    """

    def __init__(self, raw_path: Path, processed_path: Path):
        """
        Inicializa o extrator configurando o motor de processamento.
        
        Args:
            raw_path (Path): Diretório com PDFs técnicos brutos.
            processed_path (Path): Diretório para saída do dataset processado.
        """
        self.raw_path = raw_path
        self.processed_path = processed_path
        
        # Configuração de Hardware: Otimização p/ NVIDIA GTX 1650 (4GB VRAM)
        pipeline_options = PdfPipelineOptions()
        
        # Ativação de Aceleração por Hardware (CUDA)
        if torch.cuda.is_available():
            pipeline_options.accelerator_options.device = "cuda"
            logger.info("Aceleração GPU (CUDA) ativada para extração de layout.")
        else:
            pipeline_options.accelerator_options.device = "cpu"
            logger.warning("Executando em CPU: Performance de extração será reduzida.")

        # Configurações de Layout (Abordagem similar ao LayoutLM para tabelas técnicas)
        pipeline_options.do_table_structure = True
        pipeline_options.do_ocr = True # Garante extração mesmo em PDFs digitalizados (scans)
        
        self.converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        
        # Garantia de diretório de saída
        self.processed_path.mkdir(parents=True, exist_ok=True)

    def extract_document(self, file_path: Path) -> Optional[str]:
        """
        Converte um PDF individual para Markdown estruturado.
        
        Args:
            file_path (Path): Caminho absoluto do arquivo.
            
        Returns:
            Optional[str]: Conteúdo Markdown ou None em caso de falha crítica.
        """
        try:
            logger.info(f"Processando documento: {file_path.name}")
            result = self.converter.convert(file_path)
            
            # Exportação para Markdown preservando a semântica do layout
            return result.document.export_to_markdown()
            
        except Exception as e:
            logger.error(f"Falha na extração Docling de {file_path.name}: {str(e)}")
            # Aciona o fallback se o motor principal (Docling/pdfium) falhar
            return self.fallback_extract(file_path)

    def fallback_extract(self, file_path: Path) -> Optional[str]:
        """
        Extração de backup utilizando pypdf para casos onde o Docling falha na validação.
        
        Args:
            file_path (Path): Caminho absoluto do arquivo.
            
        Returns:
            Optional[str]: Conteúdo extraído via pypdf ou None em caso de falha absoluta.
        """
        try:
            logger.info(f"Tentando extração de fallback (pypdf) para: {file_path.name}")
            reader = PdfReader(file_path)
            text_parts = []
            
            for i, page in enumerate(reader.pages):
                text_parts.append(f"## Page {i+1}\n\n" + (page.extract_text() or ""))
            
            content = "\n\n".join(text_parts)
            if content.strip():
                return content
            return None
        except Exception as e:
            logger.error(f"Fallback pypdf também falhou para {file_path.name}: {str(e)}")
            return None

    def format_to_dataset(self, content: str, source: str) -> Dict[str, str]:
        """
        Formata o Markdown extraído para o padrão Alpaca (Instruction Tuning).
        
        Args:
            content (str): Texto Markdown extraído.
            source (str): Nome do arquivo fonte para rastreabilidade.
            
        Returns:
            Dict[str, str]: Dicionário formatado para JSONL.
        """
        return {
            "instruction": "Analise o seguinte trecho de documentação técnica de engenharia de petróleo e extraia os pontos fundamentais.",
            "input": f"Documento de Origem: {source}\n\n{content}",
            "output": "Aguardando geração sintética via LLM ou validação de especialistas."
        }

    def save_results(self, entries: List[Dict[str, str]], output_name: str) -> None:
        """
        Persiste as entradas processadas em um arquivo JSONL.
        
        Args:
            entries (List[Dict[str, str]]): Lista de dicionários formatados.
            output_name (str): Nome do arquivo final.
        """
        final_path = self.processed_path / output_name
        with open(final_path, 'w', encoding='utf-8') as f:
            for item in entries:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"Dataset de conhecimento salvo com sucesso: {final_path}")

    def run(self) -> None:
        """
        Executa o pipeline completo de ETL para todos os arquivos PDF no diretório RAW.
        """
        logger.info("Iniciando Fluxo de Extração Petrodora-AI")
        
        pdf_list = list(self.raw_path.glob("*.pdf"))
        if not pdf_list:
            logger.error("Nenhum arquivo PDF encontrado para processamento.")
            return

        processed_data: List[Dict[str, str]] = []
        
        for pdf_file in pdf_list:
            markdown = self.extract_document(pdf_file)
            
            if markdown:
                entry = self.format_to_dataset(markdown, pdf_file.name)
                processed_data.append(entry)
                logger.info(f"Sucesso: {pdf_file.name} integrado ao dataset.")
            else:
                logger.warning(f"Ignorando arquivo após falha no processador e no fallback: {pdf_file.name}")

        if processed_data:
            self.save_results(processed_data, "processed_knowledge.jsonl")
            logger.info(f"Pipeline concluído. {len(processed_data)} documentos processados.")
        else:
            logger.warning("Pipeline finalizado sem dados válidos para salvar.")

        # Limpeza explícita para evitar erros de encerramento do pdfium
        logger.info("Limpando recursos de extração...")
        del self.converter
        gc.collect()

if __name__ == "__main__":
    # Resolução de caminhos baseada na raiz do projeto
    ROOT_DIR = Path(__file__).parent.parent
    INPUT_DIR = ROOT_DIR / "data" / "raw"
    OUTPUT_DIR = ROOT_DIR / "data" / "processed"

    extractor = PetrodoraExtractor(INPUT_DIR, OUTPUT_DIR)
    extractor.run()
