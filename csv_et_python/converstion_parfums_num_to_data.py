import sqlite3
import pandas as pd


csv = "parfums_numerotes.csv"  
csv2 = pd.read_csv(csv)
conn = sqlite3.connect("parfums_numerotes.db")
csv2.to_sql("parfums_numerotes", conn, if_exists="replace", index=False)
conn.close()

