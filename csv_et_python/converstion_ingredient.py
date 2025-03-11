import sqlite3
import pandas as pd


csv = "ingredients.csv"  
csv2 = pd.read_csv(csv)
conn = sqlite3.connect("ingredients.db")
csv2.to_sql("ingredients", conn, if_exists="replace", index=False)
conn.close()
