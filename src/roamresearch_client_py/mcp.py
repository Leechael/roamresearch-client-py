from dotenv import load_dotenv
from fastapi import FastAPI
from mcp.server.fastmcp import FastMCP
import httpx
import pendulum
from .RoamClient import RoamClient

mcp = FastMCP(name="EchoServer", stateless_http=True)


@mcp.tool(description="Save a text block into Roam Research's Daily Notes")
async def save_block(message: str) -> str:
    async with RoamClient() as client:
        await client.write(message)
    return f"Saved"


app = FastAPI()
app.mount("/roam", mcp.streamable_http_app())


if __name__ == "__main__":
    load_dotenv()
    mcp.run("sse")
