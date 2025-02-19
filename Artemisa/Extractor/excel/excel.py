import pandas as pd

class ExcelExtractor:
    def __init__(self, path_to_excel: str):
        self.path = path_to_excel

    def excel(self):
        excel_file = pd.ExcelFile(self.path)
        output_files = []  # Lista para guardar los nombres de los archivos CSV generados
        
        for sheet_name in excel_file.sheet_names:
            df_temp = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)

            # Elimina columnas y filas completamente vacías
            df_temp.dropna(axis='columns', how='all', inplace=True)
            df_temp.dropna(axis='rows', how='all', inplace=True)

            # Genera un nombre de archivo a partir del nombre de la hoja
            output_csv = f"{sheet_name}.csv"
            df_temp.to_csv(output_csv, index=False, header=False)
            print(f"Hoja '{sheet_name}' guardada en '{output_csv}'")
            output_files.append(output_csv)
            
        return output_files, df_temp  # Retorna la lista de archivos CSV generados
