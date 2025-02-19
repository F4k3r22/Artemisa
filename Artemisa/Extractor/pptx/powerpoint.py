from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.enum.dml import MSO_FILL
import os
from PIL import Image
from io import BytesIO

class PPTXExtractor:
    def __init__(self, pptx_path):
        self.path = pptx_path
        self.presentation = Presentation(pptx_path)

    def extract_all(self):
        """Extrae todo el contenido de la presentación"""
        content = {
            'slides': self.extract_slides(),
            'properties': self.extract_properties(),
            'presentation_stats': self.get_presentation_stats()
        }
        return content

    def extract_slides(self):
        """Extrae el contenido de todas las diapositivas"""
        slides = []
        for slide_number, slide in enumerate(self.presentation.slides, 1):
            slide_content = {
                'slide_number': slide_number,
                'slide_layout': slide.slide_layout.name,
                'shapes': self.extract_shapes(slide),
                'notes': self.extract_notes(slide),
                'background': self.get_background_info(slide)
            }
            slides.append(slide_content)
        return slides

    def extract_shapes(self, slide):
        """Extrae todas las formas de una diapositiva"""
        shapes = []
        for shape in slide.shapes:
            shape_data = {
                'shape_type': str(shape.shape_type),
                'shape_name': shape.name,
                'id': shape.shape_id,
                'position': {
                    'left': shape.left,
                    'top': shape.top,
                    'width': shape.width,
                    'height': shape.height
                }
            }

            # Extraer texto si la forma lo contiene
            if hasattr(shape, "text"):
                shape_data['text'] = shape.text
                if hasattr(shape, "text_frame"):
                    shape_data['paragraphs'] = self.extract_paragraphs(shape)

            # Extraer información específica según el tipo de forma
            if shape.shape_type == MSO_SHAPE_TYPE.TABLE:
                shape_data['table_data'] = self.extract_table(shape)
            elif shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                shape_data['image_info'] = self.extract_image_info(shape)
            elif shape.shape_type == MSO_SHAPE_TYPE.CHART:
                shape_data['chart_data'] = self.extract_chart_data(shape)

            shapes.append(shape_data)
        return shapes

    def extract_paragraphs(self, shape):
        """Extrae párrafos con formato de una forma"""
        paragraphs = []
        if hasattr(shape, "text_frame"):
            for paragraph in shape.text_frame.paragraphs:
                para_data = {
                    'text': paragraph.text,
                    'level': paragraph.level,
                    'alignment': str(paragraph.alignment),
                    'runs': []
                }
                # Extraer información de cada run (fragmento de texto con formato)
                for run in paragraph.runs:
                    run_data = {
                        'text': run.text,
                        'bold': run.font.bold,
                        'italic': run.font.italic,
                        'underline': run.font.underline,
                        'font_name': run.font.name,
                        'font_size': run.font.size if run.font.size else None,
                        'color': str(run.font.color.rgb) if run.font.color.rgb else None
                    }
                    para_data['runs'].append(run_data)
                paragraphs.append(para_data)
        return paragraphs

    def extract_table(self, shape):
        """Extrae datos de una tabla"""
        table_data = []
        if shape.has_table:
            for row in shape.table.rows:
                row_data = []
                for cell in row.cells:
                    cell_data = {
                        'text': cell.text,
                        'paragraphs': self.extract_paragraphs(cell)
                    }
                    row_data.append(cell_data)
                table_data.append(row_data)
        return table_data

    def extract_image_info(self, shape):
        """Extrae información de una imagen"""
        if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
            return {
                'filename': shape.image.filename,
                'content_type': shape.image.content_type,
                'size': shape.image.size
            }
        return None

    def extract_chart_data(self, shape):
        """Extrae datos de un gráfico"""
        if shape.has_chart:
            chart_data = {
                'chart_type': str(shape.chart.chart_type),
                'has_legend': shape.chart.has_legend,
                'series': []
            }
            for series in shape.chart.series:
                series_data = {
                    'name': series.name,
                    'values': list(series.values)
                }
                if hasattr(series, 'categories'):
                    series_data['categories'] = list(series.categories)
                chart_data['series'].append(series_data)
            return chart_data
        return None

    def extract_notes(self, slide):
        """Extrae las notas de una diapositiva"""
        if slide.has_notes_slide:
            notes_slide = slide.notes_slide
            return {
                'text': notes_slide.notes_text_frame.text,
                'paragraphs': [shape.text for shape in notes_slide.shapes if hasattr(shape, "text")]
            }
        return None

    def get_background_info(self, slide):
        """Obtiene información del fondo de la diapositiva, incluyendo imágenes"""
        background = {
            'type': None,  # Puede ser: 'solid', 'picture', 'pattern', 'gradient', None
            'details': {}
        }
        
        try:
            if slide.background and slide.background.fill:
                fill = slide.background.fill
                
                # Determinar el tipo de relleno
                if fill.type == MSO_FILL.SOLID:
                    background['type'] = 'solid'
                    try:
                        if fill.solid and fill.solid.fore_color:
                            background['details']['color'] = str(fill.solid.fore_color.rgb)
                    except (AttributeError, TypeError):
                        pass

                elif fill.type == MSO_FILL.PICTURE:
                    background['type'] = 'picture'
                    try:
                        if hasattr(fill, 'picture') and fill.picture:
                            background['details'].update({
                                'filename': getattr(fill.picture, 'filename', None),
                                'content_type': getattr(fill.picture, 'content_type', None),
                                'size': getattr(fill.picture, 'size', None)
                            })
                    except (AttributeError, TypeError):
                        pass

                elif fill.type == MSO_FILL.PATTERN:
                    background['type'] = 'pattern'
                    try:
                        if fill.pattern:
                            background['details']['pattern_type'] = str(fill.pattern)
                            if hasattr(fill.pattern, 'fore_color'):
                                background['details']['fore_color'] = str(fill.pattern.fore_color.rgb)
                            if hasattr(fill.pattern, 'back_color'):
                                background['details']['back_color'] = str(fill.pattern.back_color.rgb)
                    except (AttributeError, TypeError):
                        pass

                elif fill.type == MSO_FILL.GRADIENT:
                    background['type'] = 'gradient'
                    try:
                        if fill.gradient:
                            background['details']['gradient_stops'] = []
                            for stop in fill.gradient.gradient_stops:
                                background['details']['gradient_stops'].append({
                                    'position': stop.position,
                                    'color': str(stop.color.rgb) if stop.color else None
                                })
                    except (AttributeError, TypeError):
                        pass

                # También verificamos si hay una forma de fondo
                background_shapes = [
                    shape for shape in slide.shapes 
                    if shape.is_placeholder and shape.placeholder_format.type == 14  # 14 es el tipo para fondos
                ]
                if background_shapes:
                    background['background_shape'] = True
                    if background_shapes[0].shape_type == MSO_SHAPE_TYPE.PICTURE:
                        background['type'] = 'picture_shape'
                        try:
                            picture = background_shapes[0]
                            background['details'].update({
                                'shape_name': picture.name,
                                'shape_id': picture.shape_id,
                                'width': picture.width,
                                'height': picture.height
                            })
                        except (AttributeError, TypeError):
                            pass

        except Exception as e:
            background['error'] = str(e)
            
        return background

    def extract_properties(self):
        """Extrae las propiedades de la presentación"""
        core_props = self.presentation.core_properties
        return {
            'author': core_props.author,
            'created': core_props.created,
            'modified': core_props.modified,
            'title': core_props.title,
            'subject': core_props.subject,
            'keywords': core_props.keywords,
            'last_modified_by': core_props.last_modified_by,
            'revision': core_props.revision
        }

    def get_presentation_stats(self):
        """Obtiene estadísticas generales de la presentación"""
        total_shapes = sum(len(slide.shapes) for slide in self.presentation.slides)
        return {
            'slide_count': len(self.presentation.slides),
            'total_shapes': total_shapes,
            'slide_width': self.presentation.slide_width,
            'slide_height': self.presentation.slide_height
        }

    def save_images(self, output_dir):
        """Guarda todas las imágenes encontradas en la presentación"""
        os.makedirs(output_dir, exist_ok=True)
        image_count = 0
        
        for slide in self.presentation.slides:
            for shape in slide.shapes:
                if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                    image_count += 1
                    image_bytes = shape.image.blob
                    image = Image.open(BytesIO(image_bytes))
                    image_path = os.path.join(output_dir, f'image_{image_count}.{shape.image.ext}')
                    image.save(image_path)