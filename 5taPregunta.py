import pyodbc
import pandas as pd

# Configuración de la conexión
server = 'localhost'
database = 'colegio'
username = 'sa'
password = '123456'

conn_str = (
    f'DRIVER={{ODBC Driver 17 for SQL Server}};'
    f'SERVER={server};'
    f'DATABASE={database};'
    f'UID={username};'
    f'PWD={password};'
)

try:
    conn = pyodbc.connect(conn_str)
    print("Conexión establecida correctamente.")

    query = "SELECT TOP 5 * FROM Docentes"
    df = pd.read_sql(query, conn)
    print(df)

except Exception as e:
    print(f"Error al conectar a la base de datos: {str(e)}")



def buscar_productos_por_categoria(conn, categoria):
    try:
        query = """
            SELECT Nombre, Price 
            FROM Productos 
            WHERE category = ?
        """


        df = pd.read_sql(query, conn, params=[categoria])

        if not df.empty:
            print(f"Productos en la categoría '{categoria}':")
            print(df)
            
            min_price = df['Price'].min()
            max_price = df['Price'].max()
            avg_price = df['Price'].mean()

            print(f"\nEstadísticas de precios:")
            print(f"Precio mínimo: {min_price}")
            print(f"Precio máximo: {max_price}")
            print(f"Precio promedio: {avg_price}")
        else:
            print(f"No se encontraron productos en la categoría '{categoria}'.")

    except Exception as e:
        print(f"Error al ejecutar la consulta SQL: {str(e)}")


buscar_productos_por_categoria(conn, 'Electrónicos')
