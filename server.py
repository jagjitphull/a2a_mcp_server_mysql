#Create this file in mcp_server/server.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mysql.connector
import os
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_NAME = os.getenv("DB_NAME", "salesdb")
DB_USER = os.getenv("DB_USER", "readonly")
DB_PASS = os.getenv("DB_PASS", "readonlypass")

app = FastAPI(title="MCP Server for MySQL")

# ------------------ Schemas ------------------
class MCPRequest(BaseModel):
    id: str
    method: str
    params: dict

class MCPResponse(BaseModel):
    id: str
    result: dict | None = None
    error: str | None = None

# ------------------ Helpers ------------------
def get_conn():
    return mysql.connector.connect(
        host=DB_HOST, port=DB_PORT,
        user=DB_USER, password=DB_PASS, database=DB_NAME
    )

def enforce_select_only(query: str):
    q = (query or "").strip().lower()
    if not q.startswith("select"):
        raise HTTPException(400, "Only SELECT queries allowed")
    if ";" in q:
        raise HTTPException(400, "Multiple statements not allowed")

# ------------------ Methods ------------------
def list_tables():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SHOW TABLES;")
            return [r[0] for r in cur.fetchall()]

def describe_table(table: str):
    with get_conn() as conn:
        with conn.cursor(dictionary=True) as cur:
            cur.execute(f"DESCRIBE `{table}`;")
            return cur.fetchall()

def run_select(query: str, params=None, limit=100, offset=0):
    enforce_select_only(query)
    sql = query.strip()
    if "limit" not in sql.lower():
        sql = f"{sql} LIMIT %s OFFSET %s"
        params = (params or []) + [limit, offset]
    with get_conn() as conn:
        with conn.cursor(dictionary=True) as cur:
            cur.execute(sql, params or [])
            return cur.fetchall()

# ------------------ API ------------------
@app.post("/mcp", response_model=MCPResponse)
def handle(req: MCPRequest):
    try:
        if req.method == "db.list_tables":
            result = {"tables": list_tables()}
        elif req.method == "db.describe_table":
            result = {"columns": describe_table(req.params.get("table"))}
        elif req.method == "db.select":
            result = {
                "rows": run_select(
                    req.params.get("query"),
                    req.params.get("params"),
                    req.params.get("limit", 100),
                    req.params.get("offset", 0)
                )
            }
        else:
            return MCPResponse(id=req.id, error=f"Unknown method {req.method}")
        return MCPResponse(id=req.id, result=result)
    except Exception as e:
        return MCPResponse(id=req.id, error=str(e))
