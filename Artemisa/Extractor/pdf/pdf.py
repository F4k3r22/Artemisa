import fitz  
from collections import defaultdict
import re
from typing import Dict, List, Any, Tuple

class PDFExtractor:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        # Caché para imágenes extraídas por xref (evita extraer la misma imagen varias veces)
        self._image_cache: Dict[int, Dict[str, Any]] = {}

    def extract_all(self) -> Dict:
        """Extrae todo el contenido del PDF incluyendo texto, tablas, imágenes y metadatos"""
        try:
            doc = fitz.open(self.pdf_path)
            content = {
                'metadata': dict(doc.metadata or {}),
                'total_pages': doc.page_count,
                'pages': self.extract_pages(doc),
                'tables': self.extract_all_tables(doc),
                'images': self.extract_images_info(doc),
                'text_statistics': self.get_text_statistics(doc)
            }
            doc.close()
            return content
        except Exception as e:
            raise Exception(f"Error extracting PDF content: {str(e)}")

    # ---------------- PÁGINAS ----------------
    def extract_pages(self, doc: fitz.Document) -> List[Dict]:
        """Extrae el contenido de cada página con su estructura (misma salida que tu versión)"""
        pages_content: List[Dict] = []

        # Recorremos por índices para un acceso más rápido y controlado
        for page_idx in range(doc.page_count):
            page = doc.load_page(page_idx)

            # Extracción en una sola pasada lógica:
            # - texto ordenado (sort=True mejora el orden de lectura)
            # - palabras con bbox (get_text("words")) para layout/áreas
            page_text = page.get_text("text", sort=True) or ""
            words_raw: List[Tuple] = page.get_text("words") or []

            # convertir words a dicts (coincidente con estructura previa)
            words = [
                {
                    'x0': w[0],
                    'top': w[1],
                    'x1': w[2],
                    'bottom': w[3],
                    'text': w[4]
                }
                for w in words_raw
            ]

            page_content = {
                'page_number': page_idx + 1,
                'text': page_text,
                'tables': self.extract_tables_from_text(page_text),
                'images': self.get_page_images(doc, page_idx),
                'dimensions': {
                    'width': float(page.rect.width),
                    'height': float(page.rect.height)
                },
                'layout': self.analyze_page_layout_from_words(page, words)
            }
            pages_content.append(page_content)

        return pages_content

    # ---------------- TABLAS ----------------
    def extract_tables(self, page: fitz.Page) -> List[List[List[str]]]:
        """Firma mantenida: recibe page (fitz.Page) y devuelve tablas (heurística rápida)"""
        try:
            raw_text = page.get_text("text", sort=True) or ""
        except Exception:
            raw_text = ""
        return self.extract_tables_from_text(raw_text)

    def extract_tables_from_text(self, raw_text: str) -> List[List[List[str]]]:
        """
        Heurística ligera para tablas basada en líneas:
        - detecta líneas con '|' o con múltiples espacios (p.ej. separación por columnas)
        - es rápida y suele funcionar con tablas generadas digitalmente
        """
        if not raw_text:
            return []
        tables: List[List[List[str]]] = []
        lines = raw_text.splitlines()
        candidate: List[str] = []

        def is_table_line(l: str) -> bool:
            if '|' in l:
                return True
            # 2+ espacios consecutivos suele indicar separación de columnas en texto plano
            return len(re.findall(r' {2,}', l)) >= 1

        for ln in lines:
            if is_table_line(ln):
                candidate.append(ln)
            else:
                if candidate:
                    parsed = []
                    for row in candidate:
                        if '|' in row:
                            cols = [c.strip() for c in row.split('|')]
                        else:
                            cols = [c.strip() for c in re.split(r' {2,}', row)]
                        parsed.append([str(c) if c is not None else '' for c in cols])
                    if parsed:
                        tables.append(parsed)
                    candidate = []
        # final block
        if candidate:
            parsed = []
            for row in candidate:
                if '|' in row:
                    cols = [c.strip() for c in row.split('|')]
                else:
                    cols = [c.strip() for c in re.split(r' {2,}', row)]
                parsed.append([str(c) if c is not None else '' for c in cols])
            if parsed:
                tables.append(parsed)

        return tables

    def extract_all_tables(self, doc: fitz.Document) -> List[Dict]:
        """Extrae todas las tablas del documento con su ubicación (rápido, una pasada)"""
        all_tables: List[Dict] = []
        for page_idx in range(doc.page_count):
            page = doc.load_page(page_idx)
            text = page.get_text("text", sort=True) or ""
            tables = self.extract_tables_from_text(text)
            if tables:
                all_tables.append({
                    'page_number': page_idx + 1,
                    'tables': tables,
                    'count': len(tables)
                })
        return all_tables

    # ---------------- IMÁGENES ----------------
    def extract_images_info(self, doc: fitz.Document) -> List[Dict]:
        """Extrae información sobre las imágenes en el PDF (por página)"""
        images_info: List[Dict] = []
        # Recorremos por página y usamos get_images + get_image_rects para bbox
        for page_idx in range(doc.page_count):
            page_images = self.get_page_images(doc, page_idx)
            if page_images:
                images_info.append({
                    'page_number': page_idx + 1,
                    'images': page_images
                })
        return images_info

    def get_page_images(self, doc: fitz.Document, page_idx: int) -> List[Dict]:
        """
        Obtiene info de imágenes en una página:
        - usa page.get_images(full=True) para listar referencias (xref)
        - usa page.get_image_rects(xref) para obtener bbox donde aparece la imagen
        - usa doc.extract_image(xref) para obtener metadatos (width/height/ext/bytes)
        - cachea por xref para no extraer bytes múltiples veces
        """
        try:
            page = doc.load_page(page_idx)
            img_list = page.get_images(full=True) or []
            infos: List[Dict] = []

            # Recorremos referencias, controlando duplicados por xref
            for item in img_list:
                # item is e.g. (xref, smask, width, height, bpc, colorspace, alt, name)
                xref = item[0]
                # obtener bbox(s) donde la imagen se dibuja en esta página:
                rects = []
                try:
                    rects_with_matrix = page.get_image_rects(xref, transform=True) or []
                    for (rect, matrix) in rects_with_matrix:
                        rects.append((rect.x0, rect.y0, rect.x1, rect.y1))
                except Exception:
                    # fallback: sin bbox
                    rects = []

                # extraer metadata de imagen (cacheada)
                img_meta = self._image_cache.get(xref)
                if img_meta is None:
                    try:
                        img_meta = doc.extract_image(xref) or {}
                    except Exception:
                        img_meta = {}
                    # almacenar en cache
                    self._image_cache[xref] = img_meta

                info = {
                    'xref': xref,
                    'bboxes': rects,
                    'width': img_meta.get('width'),
                    'height': img_meta.get('height'),
                    'ext': img_meta.get('ext'),
                    'colorspace': img_meta.get('colorspace') or img_meta.get('cs'),
                    # 'image_bytes' intentionally not included to keep estructura ligera;
                    # si quieres los bytes, los puedes recuperar de img_meta.get('image')
                }
                infos.append(info)

            return infos
        except Exception:
            return []

    # ---------------- LAYOUT / DETECCIÓN ----------------
    def analyze_page_layout_from_words(self, page: fitz.Page, words: List[Dict]) -> Dict:
        """
        Analiza disposición usando la lista de palabras ya extraída (evita re-llamadas).
        Devuelve la estructura {text_areas, margins, columns}
        """
        layout = {
            'text_areas': [],
            'margins': self.detect_margins_from_words(page, words),
            'columns': self.detect_columns_from_words(words)
        }
        if words:
            layout['text_areas'] = self.group_text_areas(words)
        return layout

    def detect_margins_from_words(self, page: fitz.Page, words: List[Dict]) -> Dict:
        """Detecta márgenes usando bboxes de palabras (rápido si ya tienes 'words')"""
        if not words:
            return {'top': 0, 'bottom': 0, 'left': 0, 'right': 0}
        left = min(float(w['x0']) for w in words)
        right = max(float(w['x1']) for w in words)
        top = min(float(w['top']) for w in words)
        bottom = max(float(w['bottom']) for w in words)
        page_width = float(page.rect.width)
        page_height = float(page.rect.height)
        return {
            'top': top,
            'bottom': page_height - bottom,
            'left': left,
            'right': page_width - right
        }

    def detect_columns_from_words(self, words: List[Dict]) -> int:
        """Detecta columnas por análisis de posiciones X de palabras (mismo algoritmo)"""
        if not words:
            return 1
        x_positions = sorted({float(w['x0']) for w in words})
        gaps = self.find_significant_gaps(x_positions)
        return max(1, len(gaps) + 1)

    def find_significant_gaps(self, positions: List[float], threshold: float = 50) -> List[float]:
        """Idéntico a tu versión — busca huecos amplios para inferir columnas"""
        if not positions:
            return []
        positions = sorted(set(positions))
        gaps: List[float] = []
        for i in range(len(positions) - 1):
            gap = positions[i + 1] - positions[i]
            if gap > threshold:
                gaps.append((positions[i] + positions[i + 1]) / 2)
        return gaps

    def group_text_areas(self, words: List[Dict]) -> List[Dict]:
        """Agrupa palabras en áreas de texto coherentes (misma firma)"""
        text_areas: List[Dict] = []
        current_area: List[Dict] = []
        for word in words:
            if current_area and self.is_new_area(current_area[-1], word):
                text_areas.append(self.create_text_area(current_area))
                current_area = []
            current_area.append(word)
        if current_area:
            text_areas.append(self.create_text_area(current_area))
        return text_areas

    def is_new_area(self, prev_word: Dict, curr_word: Dict, vertical_threshold: float = 15) -> bool:
        """Determina si es una nueva área — uso igual que la versión original"""
        try:
            return float(curr_word['top']) - float(prev_word['bottom']) > vertical_threshold
        except Exception:
            return False

    def create_text_area(self, words: List[Dict]) -> Dict:
        """Crea área de texto con bbox y texto concatenado (misma salida)"""
        try:
            bbox = (
                min(float(w['x0']) for w in words),
                min(float(w['top']) for w in words),
                max(float(w['x1']) for w in words),
                max(float(w['bottom']) for w in words)
            )
            text = ' '.join(w.get('text', '') for w in words)
            return {'bbox': bbox, 'text': text}
        except Exception:
            return {'bbox': None, 'text': ' '.join(w.get('text', '') for w in words)}

    # ---------------- ESTADÍSTICAS ----------------
    def get_text_statistics(self, doc: fitz.Document) -> Dict:
        """Obtiene estadísticas del texto en el documento (una pasada eficiente)"""
        stats = defaultdict(int)
        for page_idx in range(doc.page_count):
            page = doc.load_page(page_idx)
            text = page.get_text("text", sort=True) or ""
            if text:
                stats['total_characters'] += len(text)
                stats['total_words'] += len(text.split())
                stats['total_lines'] += text.count('\n') + 1
                stats['total_paragraphs'] += len(re.split(r'\n\s*\n', text))
        return dict(stats)
