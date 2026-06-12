"""
twilio_server.py

MCP server exposing Twilio SMS functionality (sending, listing, viewing messages).
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load .env variables manually in case they are not in the environment yet
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

server = Server("twilio")

# Read Twilio credentials
ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")


async def _get_auth_headers_and_sid() -> tuple[dict, str, str]:
    """Helper to get authentication headers, account SID, and phone number, prioritizing OAuth client credentials."""
    client_id = os.getenv("TWILIO_CLIENT_ID")
    client_secret = os.getenv("TWILIO_CLIENT_SECRET")
    sid = os.getenv("TWILIO_ACCOUNT_SID")
    token = os.getenv("TWILIO_AUTH_TOKEN")
    phone = os.getenv("TWILIO_PHONE_NUMBER")
    
    if not sid or not token:
        # Fallback to RJ Business Solutions credentials if primary is missing
        sid = os.getenv("TWILIO_RJ_ACCOUNT_SID")
        token = os.getenv("TWILIO_RJ_AUTH_TOKEN")
    
    # Try Client Credentials Grant (OAuth2) first if Client ID and Secret are provided
    if client_id and client_secret:
        try:
            async with httpx.AsyncClient() as client:
                url = "https://oauth.twilio.com/v2/token"
                data = {
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "grant_type": "client_credentials",
                }
                response = await client.post(url, data=data, timeout=5.0)
                if response.status_code == 200:
                    res_json = response.json()
                    access_token = res_json.get("access_token")
                    if access_token:
                        return {"Authorization": f"Bearer {access_token}"}, sid, phone
        except Exception:
            pass  # Fall back to Basic Auth on any error

    # Fallback to standard Basic Auth
    if sid and token:
        import base64
        credentials = f"{sid}:{token}"
        encoded = base64.b64encode(credentials.encode("utf-8")).decode("utf-8")
        return {"Authorization": f"Basic {encoded}"}, sid, phone
    
    return {}, sid or "", phone


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="send_sms",
            description="Send an SMS message to a phone number using Twilio.",
            inputSchema={
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "The destination phone number in E.164 format (e.g., +1234567890).",
                    },
                    "body": {
                        "type": "string",
                        "description": "The text content of the SMS message.",
                    },
                    "from_number": {
                        "type": "string",
                        "description": "Optional sender phone number. Defaults to the configured TWILIO_PHONE_NUMBER.",
                    },
                },
                "required": ["to", "body"],
            },
        ),
        Tool(
            name="list_sms_messages",
            description="Retrieve a list of recent sent and received SMS messages from Twilio.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of messages to retrieve (default: 10, max: 50).",
                        "default": 10,
                    }
                },
            },
        ),
        Tool(
            name="get_sms_message",
            description="Retrieve full details of a specific SMS message by SID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "message_sid": {
                        "type": "string",
                        "description": "The unique Twilio SID of the message.",
                    }
                },
                "required": ["message_sid"],
            },
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    headers, sid, phone = await _get_auth_headers_and_sid()
    if not sid or "Authorization" not in headers:
        return [
            TextContent(
                type="text",
                text="Error: Twilio credentials or authentication headers could not be resolved from environment.",
            )
        ]

    async with httpx.AsyncClient() as client:
        if name == "send_sms":
            to = arguments.get("to")
            body = arguments.get("body")
            from_ = arguments.get("from_number") or phone

            if not to or not body:
                return [TextContent(type="text", text="Error: both 'to' and 'body' are required.")]

            if not from_:
                return [
                    TextContent(
                        type="text",
                        text="Error: No sender phone number configured. Please specify 'from_number' or set TWILIO_PHONE_NUMBER in .env.",
                    )
                ]

            url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
            data = {
                "To": to,
                "From": from_,
                "Body": body,
            }

            try:
                response = await client.post(url, data=data, headers=headers)
                if response.status_code in (200, 201):
                    res_data = response.json()
                    msg_sid = res_data.get("sid")
                    status = res_data.get("status")
                    return [
                        TextContent(
                            type="text",
                            text=f"SMS sent successfully! SID: {msg_sid}, Status: {status}",
                        )
                    ]
                else:
                    return [
                        TextContent(
                            type="text",
                            text=f"Twilio API Error ({response.status_code}): {response.text}",
                        )
                    ]
            except Exception as e:
                return [TextContent(type="text", text=f"Request failed: {type(e).__name__}: {e}")]

        elif name == "list_sms_messages":
            limit = min(int(arguments.get("limit", 10)), 50)
            url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
            params = {"PageSize": limit}

            try:
                response = await client.get(url, params=params, headers=headers)
                if response.status_code == 200:
                    res_data = response.json()
                    messages = res_data.get("messages", [])
                    if not messages:
                        return [TextContent(type="text", text="No messages found.")]

                    lines = [f"Found {len(messages)} messages (most recent first):"]
                    for msg in messages:
                        m_sid = msg.get("sid", "")[:10]
                        m_to = msg.get("to", "")
                        m_from = msg.get("from", "")
                        m_body = msg.get("body", "")
                        m_status = msg.get("status", "")
                        m_date = msg.get("date_sent", msg.get("date_created", ""))
                        if m_body and len(m_body) > 60:
                            m_body = m_body[:57] + "..."
                        lines.append(
                            f"- [{m_date}] SID: `{msg.get('sid')}` | From: {m_from} -> To: {m_to} | Status: {m_status} | Body: \"{m_body}\""
                        )
                    return [TextContent(type="text", text="\n".join(lines))]
                else:
                    return [
                        TextContent(
                            type="text",
                            text=f"Twilio API Error ({response.status_code}): {response.text}",
                        )
                    ]
            except Exception as e:
                return [TextContent(type="text", text=f"Request failed: {type(e).__name__}: {e}")]

        elif name == "get_sms_message":
            message_sid = arguments.get("message_sid")
            if not message_sid:
                return [TextContent(type="text", text="Error: 'message_sid' is required.")]

            url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages/{message_sid}.json"

            try:
                response = await client.get(url, headers=headers)
                if response.status_code == 200:
                    msg = response.json()
                    lines = [
                        f"Message Details for SID: {message_sid}",
                        f"  From: {msg.get('from')}",
                        f"  To: {msg.get('to')}",
                        f"  Date Created: {msg.get('date_created')}",
                        f"  Date Sent: {msg.get('date_sent')}",
                        f"  Status: {msg.get('status')}",
                        f"  Direction: {msg.get('direction')}",
                        f"  Price: {msg.get('price')} {msg.get('price_unit')}",
                        f"  Body:",
                        f"    {msg.get('body')}",
                    ]
                    return [TextContent(type="text", text="\n".join(lines))]
                else:
                    return [
                        TextContent(
                            type="text",
                            text=f"Twilio API Error ({response.status_code}): {response.text}",
                        )
                    ]
            except Exception as e:
                return [TextContent(type="text", text=f"Request failed: {type(e).__name__}: {e}")]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def run():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(run())
