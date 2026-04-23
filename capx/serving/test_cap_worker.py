"""Test script for CapWorker - simulates an Agent server.

This script creates a WebSocket server that:
1. Receives query_model messages from CapWorker
2. Calls LLM service via OpenAI API
3. Sends responses back to CapWorker

Usage::

    uv run --no-sync --active python capx/serving/test_cap_worker.py \\
        --config-path env_configs/cube_stack/franka_robosuite_cube_stack.yaml
"""

from __future__ import annotations
import asyncio
import json
import logging
from click.core import F
import uvicorn
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import tyro
import websockets
from websockets.asyncio.server import serve as ws_serve

from capx.envs.launch import LaunchArgs
from capx.serving.openrouter_server import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    Message,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI argument dataclass
# ---------------------------------------------------------------------------


@dataclass
class TestAgentArgs(LaunchArgs):
    """Command-line arguments for TestAgent server."""

    # WebSocket server configuration
    agent_host: str = "localhost"
    agent_port: int = 8765
    http_port: int = 8110
    agent_id: str = "test-agent-server"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TestAgentConfig:
    """Configuration for TestAgent server."""

    args: TestAgentArgs
    """Command-line arguments."""

    config: dict[str, Any] = None
    """Configuration dictionary loaded from YAML."""

    def __post_init__(self):
        if self.config is None:
            self.config = {}


class TestAgentServer:
    """WebSocket server that simulates an Agent for testing CapWorker.

    This server:
    1. Listens for WebSocket connections
    2. Receives query_model messages
    3. Calls LLM service via OpenAI API and returns results
    """

    def __init__(self, config: TestAgentConfig):
        """Initialize TestAgentServer with configuration.

        Args:
            config: Server configuration including args and config_dict
        """
        from openai import AsyncOpenAI
        from capx.serving.openrouter_server import _load_api_keys

        self.config = config
        self.args = config.args
        self.agent_config = config.config
        self.clients = {}  # Track connected clients by agent_id
        self.server = None

        # Async message pipes for communication between HTTP and WebSocket
        self.inbound_messages = asyncio.Queue()
        self.outbound_messages = asyncio.Queue()

        # Initialize OpenAI client
        api_key = _load_api_keys(".openrouterkey")[0]
        default_headers = {
            "HTTP-Referer": "https://github.com/nvidia-gear/CaP-X",
            "X-Title": "CaP-X",
        }
        self.llm_client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            default_headers=default_headers,
        )

        logger.info(
            f"TestAgentServer initialized, will listen on {config.args.agent_host}:{config.args.agent_port}"
        )

    async def handle_client(self, websocket):
        """Handle a single WebSocket client connection.

        Args:
            websocket: WebSocket connection object
        """
        agent_id = None

        try:
            # Extract agent_id from headers
            headers = (
                dict(websocket.request.headers) if hasattr(websocket, "request") else {}
            )
            agent_id = headers.get("agent_id", "unknown-client")
            client_type = headers.get("client_type", "unknown")
            cap_tag = headers.get("cap_tag", "")

            logger.info(
                f"✓ Client {agent_id} connected (type: {client_type}, tag: {cap_tag})"
            )

            # Register client
            self.clients[agent_id] = websocket
            logger.info(
                f"Client {agent_id} registered. Total clients: {len(self.clients)}"
            )

            # Main message loop
            async for message_data in websocket:
                try:
                    message = json.loads(message_data)
                    msg_type = message.get("type", "")
                    logger.info(f"Received message from {agent_id}: {msg_type}")

                    if msg_type == "extern_tools":
                        await websocket.send(
                            json.dumps(
                                {
                                    "type": "cap_task",
                                    "content": "Pick up the green cube and gently stack it on top of the red cube, then release it.",
                                    "args": {
                                        "server_url": "{}:{}".format(
                                            self.args.agent_host, self.args.http_port
                                        )
                                    },
                                }
                            )
                        )
                    else:
                        logger.warning(f"Unknown message type: {msg_type}")

                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse message: {e}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    import traceback

                    traceback.print_exc()

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {agent_id} disconnected")
        except Exception as e:
            logger.error(f"Error handling client {agent_id}: {e}")
            import traceback

            traceback.print_exc()
        finally:
            # Unregister client
            if agent_id and agent_id in self.clients:
                del self.clients[agent_id]
                logger.info(
                    f"Client {agent_id} unregistered. Remaining clients: {len(self.clients)}"
                )

    async def _handle_query_model(self):
        """Handle query_model request from CapWorker.

        Args:
            websocket: WebSocket connection
            prompt: Prompt messages for the LLM
        """

        data = await self.inbound_messages.get()
        payload = data["content"]

        try:
            response = await self.llm_client.chat.completions.create(**payload)
            results = {"type": "chat_response"}
            try:
                results["content"] = response.choices[0].message.content
            except (KeyError, IndexError) as exc:
                raise RuntimeError(f"Unexpected response format: {response}") from exc
            # await websocket.send(json.dumps(results))
            await self.outbound_messages.put(results)
            logger.info("Query model response sent to client")
        except Exception as e:
            logger.error(f"Error handling query_model: {e}")
            import traceback

            traceback.print_exc()

            # Send error response
            await self.outbound_messages.put(
                {"type": "chat_response", "content": "", "error": str(e)}
            )

    def create_app(self) -> FastAPI:
        app = FastAPI(title="TestCap Proxy", version="1.0.0")

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.post("/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            try:
                client_kwargs = request.model_dump(exclude_none=True)
                model = client_kwargs.get("model", "")
                if model.startswith("openrouter/"):
                    client_kwargs["model"] = model[len("openrouter/") :]
                # Put request into inbound queue
                await self.inbound_messages.put(
                    {"type": "chat_completion", "content": client_kwargs}
                )
                logger.info("Chat completion request put into inbound queue")
                await self._handle_query_model()
                # Wait for response from outbound queue
                response_data = await self.outbound_messages.get()
                logger.info("Received response from outbound queue")
                if response_data.get("error"):
                    raise HTTPException(status_code=500, detail=response_data["error"])

                # Build response
                choice = ChatCompletionResponseChoice(
                    index=0,
                    message=Message(role="assistant", content=response_data["content"]),
                    finish_reason=response_data.get("finish_reason", "stop"),
                )

                return ChatCompletionResponse(
                    id=response_data.get("id", ""),
                    created=response_data.get("created", 0),
                    model=response_data.get("model", ""),
                    choices=[choice],
                )

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error in chat_completions: {e}")
                import traceback

                traceback.print_exc()
                raise HTTPException(status_code=500, detail=str(e))

        @app.get("/health")
        async def health():
            return {"status": "ok"}

        return app

    async def start(self):
        """Start WebSocket server."""

        # Create WebSocket server
        host = self.args.agent_host
        port = self.args.agent_port
        self.server = await ws_serve(self.handle_client, host, port)

        # Create and start HTTP server asynchronously (non-blocking)
        self.llm_app = self.create_app()
        config = uvicorn.Config(
            self.llm_app, host=host, port=self.args.http_port, log_level="info"
        )
        self.llm_server = uvicorn.Server(config)

        # Start HTTP server in background task
        asyncio.create_task(self.llm_server.serve())

        # Start message processor task
        asyncio.create_task(self._process_messages())

        logger.info(f"✓ WebSocket server started on ws://{host}:{port}")
        logger.info(f"✓ HTTP server started on http://{host}:8110")
        logger.info("✓ TestAgentServer is ready!")

        # Keep server running
        try:
            await self.server.wait_closed()
        except KeyboardInterrupt:
            logger.info("Server interrupted by user")
        finally:
            await self.stop()

    async def stop(self):
        """Stop the WebSocket server."""
        if self.server:
            logger.info("Shutting down TestAgentServer...")
            self.server.close()
            await self.server.wait_closed()
            logger.info("TestAgentServer stopped")

        # Stop HTTP server
        if hasattr(self, "llm_server") and self.llm_server:
            self.llm_server.should_exit = True
            logger.info("HTTP server stopped")

    async def _process_messages(self):
        """Process messages from inbound queue and put results to outbound queue."""
        logger.info("Message processor started")

        while True:
            try:
                # Get message from inbound queue
                message = await self.inbound_messages.get()
                msg_type = message.get("type", "chat_completion")

                if msg_type == "chat_completion":
                    # Handle chat_completion request
                    client_kwargs = message["client_kwargs"]

                    # Strip the "openrouter/" prefix if present
                    model = client_kwargs.get("model", "")
                    if model.startswith("openrouter/"):
                        client_kwargs["model"] = model[len("openrouter/") :]

                    client_kwargs["stream"] = False
                    response = await self.llm_client.chat.completions.create(
                        **client_kwargs
                    )

                    # Build response data
                    choices_data = []
                    for c in response.choices:
                        choices_data.append(
                            {
                                "index": c.index,
                                "message": {
                                    "role": c.message.role,
                                    "content": c.message.content,
                                },
                                "finish_reason": c.finish_reason,
                            }
                        )

                    response_data = {
                        "id": response.id,
                        "created": response.created,
                        "model": response.model,
                        "choices": choices_data,
                    }

                    # Put response into outbound queue
                    await self.outbound_messages.put(response_data)
                    logger.info("Chat completion response put into outbound queue")

            except Exception as e:
                logger.error(f"Error processing message: {e}")
                import traceback

                traceback.print_exc()

                # Put error into outbound queue if it's a chat_completion request
                if msg_type == "chat_completion":
                    await self.outbound_messages.put({"error": str(e)})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(args: TestAgentArgs) -> None:
    """Load config and start TestAgentServer."""
    from capx.utils.launch_utils import _load_config

    # Load environment configuration (needed for model settings)
    _, config, _ = _load_config(args)
    if config.get("model"):
        args.model = config["model"]
    if config.get("visual_differencing_model"):
        args.visual_differencing_model = config["visual_differencing_model"]

    # Create TestAgentConfig
    server_config = TestAgentConfig(args=args, config=config)
    # Create and start TestAgentServer
    server = TestAgentServer(server_config)

    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        logger.info("Shutting down TestAgentServer...")
    except Exception as e:
        logger.error(f"TestAgentServer failed: {e}")
        raise


if __name__ == "__main__":
    main(tyro.cli(TestAgentArgs))
