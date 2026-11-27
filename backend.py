
import sqlite3
from datetime import date
from pathlib import Path

# Database file will be created in the project folder
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "finance.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database():
    conn = get_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_type TEXT NOT NULL,
            amount REAL NOT NULL,
            category TEXT NOT NULL,
            description TEXT,
            transaction_date TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def add_transaction(transaction_type, amount, category,
                    description="", transaction_date=None):

    transaction_type = transaction_type.strip().capitalize()
    category = category.strip()

    if transaction_type not in ("Income", "Expense"):
        raise ValueError("Type must be income or expense")

    if amount <= 0:
        raise ValueError("Amount must be greater than zero")

    if not category:
        raise ValueError("Category cannot be empty")

    if transaction_date is None:
        transaction_date = date.today().isoformat()

    conn = get_connection()

    conn.execute(
        """
        
    INSERT INTO transactions
     (type, amount, category, description, transaction_date)
      VALUES (?, ?, ?, ?, ?)
        """,
        (transaction_type, amount, category,
         description, transaction_date)
    )
 

    conn.commit()
    conn.close()


def get_transactions():
    conn = get_connection()

    rows = conn.execute(
        "SELECT * FROM transactions ORDER BY id DESC"
    ).fetchall()

    conn.close()

    return [dict(row) for row in rows]


def get_summary():
    conn = get_connection()

    rows = conn.execute("""
    SELECT
    type AS transaction_type,
    COALESCE(SUM(amount), 0) AS total
    FROM transactions
    GROUP BY type
    """).fetchall()

    conn.close()

    totals = {row["transaction_type"]: row["total"] for row in rows}

    income = totals.get("Income", 0)
    expenses = totals.get("Expense", 0)

    return {
        "total_income": income,
        "total_expenses": expenses,
        "balance": income - expenses
    }



    
if __name__ == "__main__":
    initialize_database()

    conn = get_connection()
    columns = conn.execute(
        "PRAGMA table_info(transactions)"
    ).fetchall()

    print("Existing columns:")
    for column in columns:
        print(column["name"])

    conn.close()
def delete_transaction(transaction_id):
    conn = get_connection()

    cursor = conn.execute(
        "DELETE FROM transactions WHERE id = ?",
        (transaction_id,)
    )

    deleted = cursor.rowcount

    conn.commit()
    conn.close()

    return deleted > 0
