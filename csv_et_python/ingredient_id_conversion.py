import sqlite3
import pandas as pd


csv = "ingredients_id_nom.csv"  
csv2 = pd.read_csv(csv)
conn = sqlite3.connect("ingredients_id_nom.db")
csv2.to_sql("ingredients_id_nom", conn, if_exists="replace", index=False)
conn.close()

