import sqlite3
import pandas as pd
from datetime import datetime

DB_PATH = "portfolio.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS portfolio (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            symbol TEXT NOT NULL,
            buy_price REAL NOT NULL,
            quantity INTEGER NOT NULL,
            date_added TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def add_trade(username: str, symbol: str, buy_price: float, quantity: int):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    date_added = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO portfolio (username, symbol, buy_price, quantity, date_added)
        VALUES (?, ?, ?, ?, ?)
    ''', (username, symbol.upper(), buy_price, quantity, date_added))
    conn.commit()
    conn.close()

def remove_trade(trade_id: int):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM portfolio WHERE id = ?', (trade_id,))
    conn.commit()
    conn.close()

def get_portfolio(username: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query('SELECT * FROM portfolio WHERE username = ?', conn, params=(username,))
    conn.close()
    return df
