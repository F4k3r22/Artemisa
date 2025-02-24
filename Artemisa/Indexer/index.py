from pathlib import Path
import sqlite3
from typing import List, Dict, Optional
import hashlib
from datetime import datetime
import pandas as pd
import json
import os
from Artemisa.Extractor import ExcelExtractor
from Artemisa.Extractor import PDFExtractor
from Artemisa.Extractor import PPTXExtractor
from Artemisa.Extractor import DocxExtractor

class LocalDocumentIndexer:
    def __init__(self, db_path: str = "document_index.db"):
        """
        Inicializa el indexador de documentos local.
        
        Args:
            db_path: Ruta donde se guardará la base de datos SQLite
        """
        self.db_path = db_path
        self._init_database()
        
    def _init_database(self):
        """Inicializa la estructura de la base de datos."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    path TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    content TEXT,
                    metadata TEXT,
                    last_modified TIMESTAMP,
                    last_indexed TIMESTAMP,
                    checksum TEXT
                )
            """)
            
            # Crear índice de búsqueda full-text
            conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS document_index USING fts5(id, content)")
            
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calcula el checksum de un archivo para detectar cambios."""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            hasher.update(f.read())
        return hasher.hexdigest()
    
    def _extract_excel_content(self, file_path: Path) -> Dict:
        """Extrae contenido de archivos Excel usando ExcelExtractor."""
        excel_extractor = ExcelExtractor(str(file_path))
        output_files, df = excel_extractor.excel()
        
        content = []
        metadata = {"sheets": {}}
        
        # Leer cada archivo CSV generado
        for csv_file in output_files:
            sheet_name = csv_file.replace('.csv', '')
            df_sheet = pd.read_csv(csv_file)
            
            # Convertir DataFrame a texto
            sheet_content = df_sheet.to_string(index=False)
            content.append(f"Sheet: {sheet_name}\n{sheet_content}")
            
            # Guardar metadata
            metadata["sheets"][sheet_name] = {
                "rows": len(df_sheet),
                "columns": list(df_sheet.columns)
            }
            
            # Limpiar archivo CSV temporal
            os.remove(csv_file)
            
        return {
            "content": "\n\n".join(content),
            "metadata": metadata
        }
    
    def _extract_powerpoint_content(self, file_path: Path) -> Dict:
        """Extrae contenido de archivos PowerPoint usando PPTXExtractor."""
        pptx_extractor = PPTXExtractor(str(file_path))
        extracted_content = pptx_extractor.extract_all()
        
        content = []
        
        # Procesar cada diapositiva
        for slide in extracted_content['slides']:
            slide_content = []
            
            # Extraer texto de todas las formas
            for shape in slide['shapes']:
                if 'text' in shape and shape['text'].strip():
                    slide_content.append(shape['text'])
                
                # Extraer texto de párrafos si existen
                if 'paragraphs' in shape:
                    for para in shape['paragraphs']:
                        slide_content.append(para['text'])
                
                # Extraer texto de tablas si existen
                if 'table_data' in shape:
                    for row in shape['table_data']:
                        row_texts = [cell['text'] for cell in row]
                        slide_content.append(' | '.join(row_texts))
            
            content.append(f"Slide {slide['slide_number']}:\n" + "\n".join(slide_content))
        
        return {
            "content": "\n\n".join(content),
            "metadata": extracted_content
        }
    
    def _extract_word_content(self, file_path: Path) -> Dict:
        """Extrae contenido de archivos Word usando DocxExtractor."""
        docx_extractor = DocxExtractor(str(file_path))
        extracted_content = docx_extractor.extract_all()
        
        content = []
        
        # Procesar párrafos
        for para in extracted_content['paragraphs']:
            if para['text'].strip():
                content.append(para['text'])
        
        # Procesar tablas
        for table in extracted_content['tables']:
            table_content = []
            for row in table:
                row_texts = [cell['text'] for cell in row]
                table_content.append(" | ".join(row_texts))
            content.append("\nTable:\n" + "\n".join(table_content))
        
        # Procesar headers
        for header in extracted_content['headers']:
            if header['text']:
                content.append(f"Header: {header['text']}")
        
        # Procesar footers
        for footer in extracted_content['footers']:
            if footer['text']:
                content.append(f"Footer: {footer['text']}")
        
        return {
            "content": "\n\n".join(content),
            "metadata": extracted_content
        }
    
    def _extract_pdf_content(self, file_path: Path) -> Dict:
        """Extrae contenido de archivos PDF usando PDFExtractor."""
        pdf_extractor = PDFExtractor(str(file_path))
        extracted_content = pdf_extractor.extract_all()
        
        content = []
        
        # Procesar cada página
        for page in extracted_content['pages']:
            page_content = []
            
            # Añadir texto principal
            if page['text']:
                page_content.append(page['text'])
            
            # Añadir contenido de tablas
            for table in page['tables']:
                table_text = []
                for row in table:
                    table_text.append(" | ".join(str(cell) for cell in row))
                if table_text:
                    page_content.append("\nTable:\n" + "\n".join(table_text))
            
            content.append(f"Page {page['page_number']}:\n" + "\n".join(page_content))
        
        return {
            "content": "\n\n".join(content),
            "metadata": extracted_content
        }
    
    def index_directory(self, directory_path: str, file_types: Optional[List[str]] = None):
        """
        Indexa todos los documentos en un directorio.
        
        Args:
            directory_path: Ruta al directorio a indexar
            file_types: Lista opcional de extensiones de archivo a indexar
                       (por defecto: ['.xlsx', '.pptx', '.docx', '.pdf'])
        """
        if file_types is None:
            file_types = ['.xlsx', '.pptx', '.docx', '.pdf']
            
        directory = Path(directory_path)
        
        for file_type in file_types:
            for file_path in directory.rglob(f"*{file_type}"):
                self.index_document(file_path)
                
    def index_document(self, file_path: Path):
        """Indexa un documento individual."""
        file_path = Path(file_path)
        checksum = self._calculate_checksum(file_path)
        
        # Verificar si el documento ya está indexado y no ha cambiado
        with sqlite3.connect(self.db_path) as conn:
            existing = conn.execute(
                "SELECT checksum FROM documents WHERE path = ?",
                (str(file_path),)
            ).fetchone()
            
            if existing and existing[0] == checksum:
                return
        
        # Extraer contenido según el tipo de archivo
        extractors = {
            '.xlsx': self._extract_excel_content,
            '.pptx': self._extract_powerpoint_content,
            '.docx': self._extract_word_content,
            '.pdf': self._extract_pdf_content
        }
        
        file_type = file_path.suffix.lower()
        if file_type not in extractors:
            raise ValueError(f"Tipo de archivo no soportado: {file_type}")
            
        extracted = extractors[file_type](file_path)
        
        # Generar ID único
        doc_id = hashlib.sha256(str(file_path).encode()).hexdigest()
        
        # Guardar en la base de datos
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO documents 
                (id, path, filename, file_type, content, metadata, last_modified, last_indexed, checksum)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                doc_id,
                str(file_path),
                file_path.name,
                file_type,
                extracted['content'],
                json.dumps(extracted['metadata']),
                datetime.fromtimestamp(file_path.stat().st_mtime),
                datetime.now(),
                checksum
            ))
            
            # Actualizar índice de búsqueda
            conn.execute("INSERT OR REPLACE INTO document_index (id, content) VALUES (?, ?)",
                       (doc_id, extracted['content']))
            
    def search(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Busca documentos que coincidan con la consulta.
        
        Args:
            query: Texto a buscar
            limit: Número máximo de resultados
            
        Returns:
            Lista de documentos que coinciden con la búsqueda
        """
        with sqlite3.connect(self.db_path) as conn:
            results = conn.execute("""
                SELECT d.* 
                FROM document_index i
                JOIN documents d ON i.id = d.id
                WHERE document_index MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (query, limit)).fetchall()
            
            documents = []
            for row in results:
                doc = {
                    "id": row[0],
                    "path": row[1],
                    "filename": row[2],
                    "file_type": row[3],
                    "content": row[4],
                    "metadata": json.loads(row[5]),
                    "last_modified": row[6],
                    "last_indexed": row[7]
                }
                documents.append(doc)
                
            return documents
            
    def reindex_all(self):
        """Reindexa todos los documentos en la base de datos."""
        with sqlite3.connect(self.db_path) as conn:
            documents = conn.execute("SELECT path FROM documents").fetchall()
            
        for doc in documents:
            self.index_document(Path(doc[0]))
            
    def get_document_count(self) -> int:
        """Retorna el número total de documentos indexados."""
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]