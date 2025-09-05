from calendar import c
import mysql.connector

from mysql.connector.pooling import PooledMySQLConnection
from mysql.connector.abstracts import MySQLConnectionAbstract

from fastapi import FastAPI


def create_users_table(
    conn: PooledMySQLConnection | MySQLConnectionAbstract,
) -> None:
    with conn.cursor() as cursor:
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS users (id INT AUTO_INCREMENT PRIMARY KEY, name VARCHAR(255))"
        )
        conn.commit()


def get_user_list(
    conn: PooledMySQLConnection | MySQLConnectionAbstract,
) -> list[dict[str, int | str]]:
    with conn.cursor() as cursor:
        cursor.execute("SELECT id, name FROM users")
        result = cursor.fetchall()

        return [{"id": el[0], "name": el[1]} for el in result]


def insert_user(
    conn: PooledMySQLConnection | MySQLConnectionAbstract, name: str
) -> None:
    with conn.cursor() as cursor:
        cursor.execute("INSERT INTO users (name) VALUES (%s)", (name,))
        conn.commit()


app = FastAPI()


conn: PooledMySQLConnection | MySQLConnectionAbstract = mysql.connector.connect(
    host="mysql", user="user", password="password", database="test_db"
)

create_users_table(conn)


@app.get("/users")
def read_users():
    return get_user_list(conn)


@app.post("/users")
def add_user(name: str):
    insert_user(conn, name)
    return {"message": f"User '{name}' added successfully."}
