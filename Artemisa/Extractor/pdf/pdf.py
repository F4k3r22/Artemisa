import pdfplumber
from collections import defaultdict
import re
from typing import Dict, List

class PDFExtractor:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        
    def extract_all(self) -> Dict:
        """Extrae todo el contenido del PDF incluyendo texto, tablas, imágenes y metadatos"""
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                content = {
                    'metadata': pdf.metadata,
                    'total_pages': len(pdf.pages),
                    'pages': self.extract_pages(pdf),
                    'tables': self.extract_all_tables(pdf),
                    'images': self.extract_images_info(pdf),
                    'text_statistics': self.get_text_statistics(pdf)
                }
                return content
        except Exception as e:
            raise Exception(f"Error extracting PDF content: {str(e)}")

    def extract_pages(self, pdf) -> List[Dict]:
        """Extrae el contenido de cada página con su estructura"""
        pages_content = []
        
        for page_num, page in enumerate(pdf.pages, 1):
            page_content = {
                'page_number': page_num,
                'text': page.extract_text(),
                'tables': self.extract_tables(page),
                'images': self.get_page_images(page),
                'dimensions': {
                    'width': page.width,
                    'height': page.height
                },
                'layout': self.analyze_page_layout(page)
            }
            pages_content.append(page_content)
            
        return pages_content

    def extract_tables(self, page) -> List[List[List[str]]]:
        """Extrae las tablas de una página"""
        tables = []
        try:
            page_tables = page.extract_tables()
            for table in page_tables:
                # Limpieza de datos de la tabla
                cleaned_table = [
                    [str(cell).strip() if cell is not None else '' for cell in row]
                    for row in table
                ]
                tables.append(cleaned_table)
        except Exception as e:
            print(f"Warning: Error extracting tables from page: {str(e)}")
        return tables

    def extract_all_tables(self, pdf) -> List[Dict]:
        """Extrae todas las tablas del documento con su ubicación"""
        all_tables = []
        for page_num, page in enumerate(pdf.pages, 1):
            tables = self.extract_tables(page)
            if tables:
                table_info = {
                    'page_number': page_num,
                    'tables': tables,
                    'count': len(tables)
                }
                all_tables.append(table_info)
        return all_tables

    def extract_images_info(self, pdf) -> List[Dict]:
        """Extrae información sobre las imágenes en el PDF"""
        images_info = []
        for page_num, page in enumerate(pdf.pages, 1):
            page_images = self.get_page_images(page)
            if page_images:
                images_info.append({
                    'page_number': page_num,
                    'images': page_images
                })
        return images_info

    def get_page_images(self, page) -> List[Dict]:
        """Obtiene información sobre las imágenes en una página"""
        images = []
        try:
            for image in page.images:
                image_info = {
                    'bbox': image.get('bbox', None),
                    'width': image.get('width', None),
                    'height': image.get('height', None),
                    'type': image.get('type', None),
                    'colorspace': image.get('colorspace', None)
                }
                images.append(image_info)
        except Exception as e:
            print(f"Warning: Error extracting images: {str(e)}")
        return images

    def analyze_page_layout(self, page) -> Dict:
        """Analiza la disposición del contenido en la página"""
        layout = {
            'text_areas': [],
            'margins': self.detect_margins(page),
            'columns': self.detect_columns(page)
        }
        
        # Detectar áreas de texto
        words = page.extract_words()
        if words:
            layout['text_areas'] = self.group_text_areas(words)
            
        return layout

    def detect_margins(self, page) -> Dict:
        """Detecta los márgenes de la página"""
        words = page.extract_words()
        if not words:
            return {'top': 0, 'bottom': 0, 'left': 0, 'right': 0}
        
        left = min(float(word['x0']) for word in words)
        right = max(float(word['x1']) for word in words)
        top = min(float(word['top']) for word in words)
        bottom = max(float(word['bottom']) for word in words)
        
        return {
            'top': top,
            'bottom': page.height - bottom,
            'left': left,
            'right': page.width - right
        }

    def detect_columns(self, page) -> int:
        """Detecta el número probable de columnas en la página"""
        words = page.extract_words()
        if not words:
            return 1
            
        x_positions = [float(word['x0']) for word in words]
        gaps = self.find_significant_gaps(x_positions)
        
        # Si hay gaps significativos, probablemente hay columnas
        return len(gaps) + 1

    def find_significant_gaps(self, positions: List[float], threshold: float = 50) -> List[float]:
        """Encuentra espacios significativos que podrían indicar columnas"""
        positions = sorted(set(positions))
        gaps = []
        
        for i in range(len(positions) - 1):
            gap = positions[i + 1] - positions[i]
            if gap > threshold:
                gaps.append((positions[i] + positions[i + 1]) / 2)
                
        return gaps

    def group_text_areas(self, words: List[Dict]) -> List[Dict]:
        """Agrupa palabras en áreas de texto coherentes"""
        text_areas = []
        current_area = []
        
        for word in words:
            if current_area and self.is_new_area(current_area[-1], word):
                text_areas.append(self.create_text_area(current_area))
                current_area = []
            current_area.append(word)
            
        if current_area:
            text_areas.append(self.create_text_area(current_area))
            
        return text_areas

    def is_new_area(self, prev_word: Dict, curr_word: Dict, 
                    vertical_threshold: float = 15) -> bool:
        """Determina si una palabra comienza una nueva área de texto"""
        return float(curr_word['top']) - float(prev_word['bottom']) > vertical_threshold

    def create_text_area(self, words: List[Dict]) -> Dict:
        """Crea un área de texto a partir de un grupo de palabras"""
        return {
            'bbox': (
                min(float(w['x0']) for w in words),
                min(float(w['top']) for w in words),
                max(float(w['x1']) for w in words),
                max(float(w['bottom']) for w in words)
            ),
            'text': ' '.join(w['text'] for w in words)
        }

    def get_text_statistics(self, pdf) -> Dict:
        """Obtiene estadísticas del texto en el documento"""
        stats = defaultdict(int)
        
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                stats['total_characters'] += len(text)
                stats['total_words'] += len(text.split())
                stats['total_lines'] += text.count('\n') + 1
                stats['total_paragraphs'] += len(re.split(r'\n\s*\n', text))
                
        return dict(stats)