import sqlite3

conn = sqlite3.connect('deepfake.db')
if conn is not None:
    print("Database connection established successfully")
c = conn.cursor()
c.execute('DROP TABLE IF EXISTS Users')
