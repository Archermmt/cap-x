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
from dataclasses import dataclass
from typing import Any

import tyro
import websockets
from websockets.asyncio.server import serve as ws_serve

from capx.envs.launch import LaunchArgs

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
    agent_id: str = "test-agent-server"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TestAgentConfig:
    """Configuration for TestAgent server."""

    args: TestAgentArgs
    """Command-line arguments."""

    config_dict: dict[str, Any] = None
    """Configuration dictionary loaded from YAML."""

    def __post_init__(self):
        if self.config_dict is None:
            self.config_dict = {}


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
        from openai import OpenAI
        from capx.serving.openrouter_server import _load_api_keys

        self.config = config
        self.args = config.args
        self.config_dict = config.config_dict
        self.clients = {}  # Track connected clients by agent_id
        self.server = None

        # Initialize OpenAI client
        api_key = _load_api_keys(".openrouterkey")[0]
        default_headers = {
            "HTTP-Referer": "https://github.com/nvidia-gear/CaP-X",
            "X-Title": "CaP-X",
        }
        self.llm_client = OpenAI(
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
            # Extract agent_id from URL query parameters
            path = websocket.request.path if hasattr(websocket, "request") else ""
            if "agent_id=" in path:
                agent_id = path.split("agent_id=")[1].split("&")[0]
            else:
                agent_id = "unknown-client"

            logger.info(f"✓ Client {agent_id} connected")

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

                    if msg_type == "query_model":
                        await self._handle_query_model(websocket, message["prompt"])
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

    async def _handle_query_model(self, websocket, payload: dict):
        """Handle query_model request from CapWorker.

        Args:
            websocket: WebSocket connection
            prompt: Prompt messages for the LLM
        """

        try:
            response = await self.llm_client.chat.completions.create(**payload)
            print(f"[TMINFO] Response: {response}", flush=True)
            results = {"type": "query_model_response"}
            try:
                results["content"] = response.choices[0].message.content
                results["reasoning"] = response.choices[0].message.reasoning
            except (KeyError, IndexError) as exc:
                raise RuntimeError(f"Unexpected response format: {response}") from exc
            await websocket.send(json.dumps(results))
            logger.info("Query model response sent to client")
        except Exception as e:
            logger.error(f"Error handling query_model: {e}")
            import traceback

            traceback.print_exc()

            # Send error response
            await websocket.send(
                json.dumps(
                    {"type": "query_model_response", "content": "", "error": str(e)}
                )
            )

    async def start(self):
        """Start WebSocket server."""

        # Create WebSocket server
        host = self.args.agent_host
        port = self.args.agent_port
        self.server = await ws_serve(self.handle_client, host, port)

        logger.info(f"✓ WebSocket server started on ws://{host}:{port}")
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(args: TestAgentArgs) -> None:
    """Load config and start TestAgentServer."""
    from capx.utils.launch_utils import _load_config

    # Load environment configuration (needed for model settings)
    _, config_dict, _ = _load_config(args)

    # Create TestAgentConfig
    server_config = TestAgentConfig(
        args=args,
        config_dict=config_dict,
    )

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
