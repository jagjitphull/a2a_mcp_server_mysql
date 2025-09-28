#Create this file in a2a_agent folder

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx, os
from dotenv import load_dotenv

load_dotenv()

MCP_URL = os.getenv("MCP_URL", "http://127.0.0.1:10110/mcp")

app = FastAPI(title="DB Tool Agent")

# ------------------ Schemas ------------------
class ToolRequest(BaseModel):
    tool_name: str
    arguments: dict

class ToolResponse(BaseModel):
    output: dict | list | str | None = None
    error: str | None = None

# ------------------ Map tools → MCP ------------------
TOOL_MAP = {
    "mysql_list_tables": "db.list_tables",
    "mysql_describe_table": "db.describe_table",
    "mysql_query": "db.select",
}

# ------------------ API ------------------
@app.post("/tools/invoke", response_model=ToolResponse)
def invoke(req: ToolRequest):
    if req.tool_name not in TOOL_MAP:
        raise HTTPException(400, f"Unknown tool {req.tool_name}")

    mcp_req = {
        "id": "1",
        "method": TOOL_MAP[req.tool_name],
        "params": req.arguments
    }

    try:
        with httpx.Client(timeout=60) as client:
            r = client.post(MCP_URL, json=mcp_req)
            r.raise_for_status()
            data = r.json()
            if "error" in data and data["error"]:
                return ToolResponse(error=data["error"])
            return ToolResponse(output=data.get("result"))
    except Exception as e:
        raise HTTPException(500, f"Failed calling MCP: {e}")
