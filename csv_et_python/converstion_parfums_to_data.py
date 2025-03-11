import sqlite3
import pandas as pd


csv = "parfums.csv" 
csv2 = pd.read_csv(csv)
conn = sqlite3.connect("parfums.db")
csv2.to_sql("parfums", conn, if_exists="replace", index=False)
conn.close()


