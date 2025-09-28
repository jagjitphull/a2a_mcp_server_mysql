 #Create this file orchestrator_llm.py in orchstrator folder

import os, json, re
from typing import Any, Dict, List, Optional, Tuple
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv
import httpx

from typing import Any,Dict,List,Optional,Union


# --- add this helper near the top (after imports) ---
def _unwrap_output(tool: str, output):
    """Normalize tool outputs to a simple list for summarize()/rows."""
    if tool == "mysql_list_tables":
        if isinstance(output, dict) and "tables" in output:
            return output["tables"]
        return output
    if tool == "mysql_describe_table":
        if isinstance(output, dict) and "columns" in output:
            return output["columns"]
        return output
    if tool == "mysql_query":
        if isinstance(output, dict) and "rows" in output:
            return output["rows"]
        return output
    return output


load_dotenv()

DB_TOOL_ENDPOINT = os.getenv("DB_TOOL_ENDPOINT", "http://127.0.0.1:10111/tools/invoke")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()

# Optional providers (import only if chosen)
openai = None
genai = None

if LLM_PROVIDER == "openai":
    try:
        from openai import OpenAI
        openai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    except Exception as e:
        raise RuntimeError(f"OpenAI selected but not available: {e}")

if LLM_PROVIDER == "vertex":
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
    except Exception as e:
        raise RuntimeError(f"Vertex selected but not available: {e}")

#OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://192.168.1.23:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:latest")

app = FastAPI(title="A2A Orchestrator (LLM-Planned)")

# ------------------ Schemas ------------------
class ChatRequest(BaseModel):
    message: str
    limit: Optional[int] = 100
    offset: Optional[int] = 0

class ToolCall(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]

class ChatResponse(BaseModel):
    decision: str
    tool_called: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = None
    summary: str
    rows: Optional[Union[List[Dict[str, Any]],List[str], List[Any]]] = None
    raw_llm: Optional[str] = None

# ------------------ Tool Registry ------------------
TOOLS = [
    {
        "name": "mysql_list_tables",
        "description": "List tables available in the sales database.",
        "schema": {}
    },
    {
        "name": "mysql_describe_table",
        "description": "Describe a table's columns. Requires {table: string}.",
        "schema": {"table": "string"}
    },
    {
        "name": "mysql_query",
        "description": "Execute a READ-ONLY query. 'SELECT ...' only. Supports {query: string, params?: any[], limit?: int, offset?: int}.",
        "schema": {"query": "string", "params": "array?", "limit": "int?", "offset": "int?"}
    }
]

# ------------------ Guardrails ------------------
def enforce_select_only(sql: str):
    q = (sql or "").strip().lower()
    if not q.startswith("select"):
        raise HTTPException(400, "Only SELECT queries are allowed.")
    if ";" in q:
        raise HTTPException(400, "Multiple statements are not allowed.")
    # Subtle write ops hidden in CTEs or comments are still blocked by MCP server;
    # we double-guard here anyway.

# ------------------ LLM Prompt ------------------
SYSTEM_PROMPT = """You are a planner that chooses ONE database tool to answer the user.
You must output STRICT JSON ONLY with this shape:
{"tool_name": "<one of mysql_list_tables|mysql_describe_table|mysql_query>", "arguments": {...}}

Rules:
- If user asks to list tables → use mysql_list_tables with {}.
- If user asks schema/columns of a table → use mysql_describe_table with {"table": "<name>"}.
- For any data retrieval → use mysql_query with a SELECT-only query.
- Use parameters whenever a user-provided value appears (e.g., WHERE city = %s) and put values in "params".
- Always include a reasonable "limit" (default 100) and "offset" (default 0) for mysql_query.
- NEVER emit anything except the JSON object. No prose, no markdown, no code block fences.
"""

FEW_SHOTS = [
    {
        "user": "list tables",
        "json": {"tool_name": "mysql_list_tables", "arguments": {}}
    },
    {
        "user": "what are the columns in orders table?",
        "json": {"tool_name": "mysql_describe_table", "arguments": {"table": "orders"}}
    },
    {
        "user": "show total paid amount by customer, highest first",
        "json": {"tool_name": "mysql_query",
                 "arguments": {
                    "query": "select c.name, sum(o.amount) as total from customers c join orders o on o.customer_id=c.id where o.status = %s group by c.name order by total desc",
                    "params": ["PAID"],
                    "limit": 100,
                    "offset": 0
                 }}
    }
]

def build_messages(user_msg: str) -> List[Dict[str, str]]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in FEW_SHOTS:
        msgs.append({"role": "user", "content": ex["user"]})
        msgs.append({"role": "assistant", "content": json.dumps(ex["json"], ensure_ascii=False)})
    msgs.append({"role": "user", "content": user_msg})
    return msgs

# ------------------ LLM Drivers ------------------
def call_openai(messages: List[Dict[str,str]]) -> str:
    # Using "Responses" or "ChatCompletions" style depends on SDK version; here we use chat.completions
    resp = openai.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=messages,
        temperature=0
    )
    return resp.choices[0].message.content.strip()

import google.generativeai as genai

def call_vertex(messages):
    # API key mode
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
    else:
        # ADC mode (running on GCP or Cloud Shell)
        # For region-aware endpoints, set model with "models/..." only when needed.
        genai.configure()  # uses ADC automatically

    model_name = os.getenv("VERTEX_MODEL", "gemini-1.5-flash-002")
    # Flatten chat messages to a single prompt for safety, or use the SDK's chat API if desired:
    prompt = "\n".join(f"{m['role'].upper()}:\n{m['content']}" for m in messages)

    model = genai.GenerativeModel(model_name)
    resp = model.generate_content(
        prompt,
        generation_config={"temperature": 0,
                           "response_mime_type":"application/json",
                           },
        request_options={"timeout": 120},
    )
    return resp.text.strip()


def _flatten_messages_to_prompt(messages):
    # Convert system + few-shots + user into one prompt for /api/generate
    # We still enforce “JSON ONLY” in the system content.
    parts = []
    for m in messages:
        role = m.get("role", "user").upper()
        parts.append(f"{role}:\n{m.get('content','').strip()}\n")
    return "\n".join(parts).strip()

def call_ollama(messages):
    chat_payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0}
    }
    with httpx.Client(timeout=120) as client:
        # 1) Try /api/chat (newer Ollama)
        r = client.post(f"{OLLAMA_BASE_URL}/api/chat", json=chat_payload)
        if r.status_code == 200:
            data = r.json()
            return data["message"]["content"].strip()

        # 2) If 404, fall back to /api/generate (older Ollama)
        if r.status_code == 404:
            prompt = _flatten_messages_to_prompt(messages)
            gen_payload = {
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0}
            }
            r2 = client.post(f"{OLLAMA_BASE_URL}/api/generate", json=gen_payload)
            if r2.status_code != 200:
                raise HTTPException(status_code=503, detail=f"Ollama /api/generate failed ({r2.status_code}): {r2.text}")
            txt = r2.json().get("response", "").strip()
            if not txt:
                raise HTTPException(status_code=503, detail="Ollama returned empty response on /api/generate.")
            return txt

        # 3) Other error from /api/chat
        raise HTTPException(status_code=503, detail=f"Ollama /api/chat failed ({r.status_code}): {r.text}")


def run_llm(messages: List[Dict[str,str]]) -> str:
    if LLM_PROVIDER == "openai":
        return call_openai(messages)
    if LLM_PROVIDER == "vertex":
        return call_vertex(messages)
    return call_ollama(messages)

# ------------------ Tool Invocation ------------------
def call_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
    # extra guard before hitting DB
    if tool_name == "mysql_query":
        enforce_select_only(arguments.get("query", ""))

    payload = {"tool_name": tool_name, "arguments": arguments}
    with httpx.Client(timeout=120) as client:
        r = client.post(DB_TOOL_ENDPOINT, json=payload)
        if r.status_code != 200:
            raise HTTPException(r.status_code, r.text)
        return r.json().get("output")

def summarize(decision: str, tool: str, args: dict, output) -> str:
    out = _unwrap_output(tool, output)

    if tool == "mysql_list_tables":
        if isinstance(out, list):
            return f"Tables: {', '.join(map(str, out))}" if out else "No tables found."
        return f"Tables: {out}"

    if tool == "mysql_describe_table":
        if isinstance(out, list) and out and isinstance(out[0], dict):
            cols = [f"{r.get('Field','?')}({r.get('Type','?')})" for r in out]
        elif isinstance(out, list):
            cols = list(map(str, out))
        else:
            cols = [str(out)]
        return "Columns: " + ", ".join(cols)

    if tool == "mysql_query":
        if isinstance(out, list):
            n = len(out)
            import json
            head = json.dumps(out[:3], ensure_ascii=False)
            return f"Rows: {n}. Preview: {head}..."
        return "Query executed."

    return "Done."


# ------------------ API ------------------
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    messages = build_messages(req.message)
    raw = run_llm(messages)

    # The LLM must output strict JSON. Validate.
    try:
        data = json.loads(raw)
        tool_call = ToolCall(**data)
    except (json.JSONDecodeError, ValidationError) as e:
        # Fallback: refuse with guidance (keep it strict to avoid unsafe runs)
        raise HTTPException(400, f"Planner did not return valid JSON tool call: {e}\nRaw LLM: {raw}")

    # inject limit/offset overrides for mysql_query
    if tool_call.tool_name == "mysql_query":
        tool_call.arguments.setdefault("limit", req.limit or 100)
        tool_call.arguments.setdefault("offset", req.offset or 0)

    output = call_tool(tool_call.tool_name, tool_call.arguments)

    unwrapped = _unwrap_output(tool_call.tool_name, output)

    return ChatResponse(
        decision="llm-plan",
        tool_called=tool_call.tool_name,
        arguments=tool_call.arguments,
        summary=summarize("llm-plan", tool_call.tool_name, tool_call.arguments, output),
        rows=unwrapped if isinstance(unwrapped, list) else None,
        raw_llm=raw
    )
