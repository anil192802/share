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
            tag TEXT DEFAULT 'HOLD',
            stop_loss REAL,
            date_added TEXT NOT NULL
        )
    ''')
    
    # Update existing table if columns are missing
    cursor.execute("PRAGMA table_info(portfolio);")
    columns = [col[1] for col in cursor.fetchall()]
    if "tag" not in columns:
        cursor.execute("ALTER TABLE portfolio ADD COLUMN tag TEXT DEFAULT 'HOLD';")
    if "stop_loss" not in columns:
        cursor.execute("ALTER TABLE portfolio ADD COLUMN stop_loss REAL;")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            is_admin BOOLEAN NOT NULL DEFAULT 0,
            session_token TEXT
        )
    ''')
    
    # Check if session_token column exists in case db was already created
    cursor.execute("PRAGMA table_info(users);")
    columns = [col[1] for col in cursor.fetchall()]
    if "session_token" not in columns:
        cursor.execute("ALTER TABLE users ADD COLUMN session_token TEXT;")
    
    # insert default admin if not exists
    cursor.execute("SELECT id FROM users WHERE username = 'anil'")
    if not cursor.fetchone():
        cursor.execute('''
            INSERT INTO users (username, password, is_admin)
            VALUES ('anil', 'intelceleron', 1)
        ''')
        
    conn.commit()
    conn.close()

def authenticate_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT is_admin FROM users WHERE username = ? AND password = ?", (username, password))
    user = cursor.fetchone()
    conn.close()
    if user:
        return True, bool(user[0])
    return False, False

def set_user_session(username, token):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET session_token = ? WHERE username = ?", (token, username))
    conn.commit()
    conn.close()

def get_user_by_session(token):
    if not token:
        return None, False
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT username, is_admin FROM users WHERE session_token = ?", (token,))
    user = cursor.fetchone()
    conn.close()
    if user:
        return user[0], bool(user[1])
    return None, False

def create_user(username, password):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, 0)", (username, password))
        conn.commit()
        conn.close()
        return True, "User created successfully"
    except sqlite3.IntegrityError:
        return False, "Username already exists"
    except Exception as e:
        return False, str(e)

def delete_user(username):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        # Delete user's portfolio trades first
        cursor.execute("DELETE FROM portfolio WHERE username = ?", (username,))
        # Delete the user
        cursor.execute("DELETE FROM users WHERE username = ?", (username,))
        conn.commit()
        conn.close()
        return True, f"User {username} deleted successfully"
    except Exception as e:
        return False, str(e)

def list_users():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT username, is_admin FROM users")
    users = cursor.fetchall()
    conn.close()
    return users

def add_trade(username: str, symbol: str, buy_price: float, quantity: int, stop_loss: float = None, tag: str = 'HOLD'):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    date_added = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO portfolio (username, symbol, buy_price, quantity, stop_loss, tag, date_added)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (username, symbol.upper(), buy_price, quantity, stop_loss, tag, date_added))
    conn.commit()
    conn.close()

def update_trade_tag(trade_id: int, tag: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('UPDATE portfolio SET tag = ? WHERE id = ?', (tag, trade_id))
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
