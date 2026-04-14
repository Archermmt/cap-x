"""Test script for CapWorker - simulates an Agent server.

This script creates a WebSocket server that:
1. Implements CapProto protocol for authentication
2. Receives messages from CapWorker (query_code, query_decision)
3. Calls LLM service via _query_model from trial.py
4. Sends responses back to CapWorker

Usage::

    uv run --no-sync --active python capx/serving/test_cap_worker.py \\
        --agent-url ws://localhost:8765/agent \\
        --config-path env_configs/cube_stack/franka_robosuite_cube_stack.yaml
"""

from __future__ import annotations

import asyncio
import json
import logging
import signal
from dataclasses import dataclass
from typing import Any

import tyro
import websockets
from websockets.asyncio.server import serve as ws_serve

from capx.envs.launch import LaunchArgs
from capx.llm.client import ModelQueryArgs, query_model as _query_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI argument dataclass
# ---------------------------------------------------------------------------


@dataclass
class TestAgentArgs(LaunchArgs):
    """Command-line arguments for TestAgent server."""

    # WebSocket server configuration
    listen_host: str = "localhost"
    """Host to listen on."""

    listen_port: int = 8765
    """Port to listen on."""

    agent_id: str = "test-agent-server"
    """Server identifier."""


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
    2. Implements CapProto authentication
    3. Receives query_code and query_decision messages
    4. Calls LLM service and returns results
    """

    def __init__(self, config: TestAgentConfig):
        """Initialize TestAgentServer with configuration.

        Args:
            config: Server configuration including args and config_dict
        """
        self.config = config
        self.args = config.args
        self.config_dict = config.config_dict
        self.clients = {}  # Track connected clients by agent_id
        self.server = None

        logger.info(
            f"TestAgentServer initialized, will listen on {config.args.listen_host}:{config.args.listen_port}"
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

                    if msg_type == "query_code":
                        await self._handle_query_code(websocket, message)

                    elif msg_type == "query_decision":
                        await self._handle_query_decision(websocket, message)

                    elif msg_type == "ping":
                        await websocket.send(json.dumps({"type": "pong"}))

                    elif msg_type == "task_result":
                        logger.info(
                            f"Received task result: trial={message.get('trial')}, success={message.get('success')}"
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

    async def _handle_query_code(self, websocket, message: dict[str, Any]):
        """Handle code generation query.

        Args:
            websocket: WebSocket connection
            message: Query message containing prompt
        """
        try:
            prompt = message.get("prompt", [])
            task_description = message.get("task_description", "")

            logger.info(
                f"Processing code generation query (prompt length: {len(prompt)})"
            )

            # Call LLM service
            content = await self._call_llm(prompt)

            # Extract reasoning if available
            reasoning = content.get("reasoning")
            code_content = content.get("content", "")

            # Send response
            response = {
                "type": "code_response",
                "content": code_content,
                "reasoning": reasoning,
            }

            await websocket.send(json.dumps(response))
            logger.info("Code response sent to client")

        except Exception as e:
            logger.error(f"Error handling query_code: {e}")
            import traceback

            traceback.print_exc()

            # Send error response
            await websocket.send(
                json.dumps({"type": "code_response", "content": "", "error": str(e)})
            )

    async def _handle_query_decision(self, websocket, message: dict[str, Any]):
        """Handle multi-turn decision query.

        Args:
            websocket: WebSocket connection
            message: Query message containing prompt
        """
        try:
            prompt = message.get("prompt", [])
            task_description = message.get("task_description", "")

            logger.info(f"Processing decision query (prompt length: {len(prompt)})")

            # Call LLM service
            content = await self._call_llm(prompt)

            # Extract reasoning if available
            reasoning = content.get("reasoning")
            decision_content = content.get("content", "")

            # Send response
            response = {
                "type": "decision_response",
                "content": decision_content,
                "reasoning": reasoning,
            }

            await websocket.send(json.dumps(response))
            logger.info("Decision response sent to client")

        except Exception as e:
            logger.error(f"Error handling query_decision: {e}")
            import traceback

            traceback.print_exc()

            # Send error response
            await websocket.send(
                json.dumps(
                    {"type": "decision_response", "content": "", "error": str(e)}
                )
            )

    async def _call_llm(self, prompt: list[dict]) -> dict[str, Any]:
        """Call LLM service using _query_model from trial.py.

        Args:
            prompt: Prompt messages for the LLM

        Returns:
            Dictionary with 'content' and optional 'reasoning'
        """
        # Build ModelQueryArgs from config
        model_args = ModelQueryArgs(
            model=self.args.model,
            server_url=self.args.server_url,
            api_key=self.args.api_key,
            temperature=self.args.temperature,
            max_tokens=self.args.max_tokens,
            reasoning_effort=getattr(self.args, "reasoning_effort", "medium"),
            debug=False,
        )

        logger.info(
            f"Calling LLM: model={model_args.model}, server={model_args.server_url}"
        )

        # Run synchronous _query_model in executor to avoid blocking event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: _query_model(model_args, prompt)
        )

        logger.info(
            f"LLM call completed (response length: {len(result.get('content', ''))})"
        )

        return result

    async def start(self):
        """Start the WebSocket server."""
        host = self.args.listen_host
        port = self.args.listen_port

        logger.info(f"Starting TestAgentServer on ws://{host}:{port}")

        # Create WebSocket server
        self.server = await ws_serve(
            self.handle_client,
            host,
            port,
        )

        logger.info(f"✓ TestAgentServer listening on ws://{host}:{port}")

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
