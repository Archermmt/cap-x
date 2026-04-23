"""CapWorker launcher for WebSocket-based agent interaction.

This module implements a WebSocket client that connects to an external agent
server and executes CaP-X trial logic, communicating with the agent through
WebSocket messages for code generation and multi-turn decisions.

Usage::

    uv run --no-sync --active python -m capx.serving.launch_cap_worker \\
        --agent-url ws://localhost:8765/agent \\
        --config-path env_configs/cube_stack/franka_robosuite_cube_stack.yaml
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

import tyro
from pathlib import Path
from websockets.asyncio.client import connect as ws_connect

from capx.envs.launch import LaunchArgs
from capx.envs.configs.instantiate import instantiate

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CLI argument dataclass
# ---------------------------------------------------------------------------


@dataclass
class CapWorkerArgs(LaunchArgs):
    """Command-line arguments for CapWorker.

    Extends LaunchArgs with WebSocket connection configuration for agent interaction.
    """

    # WebSocket connection configuration (CapWorker specific)
    agent_host: str = "localhost"
    """Host of the agent server to connect to."""

    agent_port: int = 8765
    """Port of the agent server to connect to."""

    http_port: int = 8112
    """http port to listen on."""

    agent_id: str = "cap-worker"
    """Agent ID to identify this worker to the server."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CapWorkerConfig:
    """Configuration for CapWorker."""

    args: CapWorkerArgs
    """Command-line arguments."""

    env_factory: Any = None
    """Environment factory function."""

    config: dict[str, Any] = None
    """Configuration dictionary loaded from YAML."""

    def __post_init__(self):
        if self.config is None:
            self.config = {}


class CapWorker:
    """WebSocket client for agent interaction with CaP-X environment.

    This worker connects to an external agent server via WebSocket and
    executes the complete trial flow, sending queries to the agent and
    receiving responses for code generation and decision making.
    """

    def __init__(self, config: CapWorkerConfig):
        """Initialize CapWorker with configuration.

        Args:
            config: Worker configuration including args, env_factory, and config_dict
        """
        self.config = config
        self.args = config.args
        self.env_factory = config.env_factory
        self.worker_config = config.config
        self.websocket = None
        self.env = None
        agent_url = f"ws://{config.args.agent_host}:{config.args.agent_port}"
        logger.info(f"CapWorker initialized, will connect to: {agent_url}")

    async def _send_message(self, message: dict[str, Any]):
        """Send a message to the agent via WebSocket.

        Args:
            message: Message dictionary to send
        """
        if self.websocket is None:
            raise RuntimeError("Not connected to agent server")

        await self.websocket.send(json.dumps(message))
        logger.debug(f"Sent to agent: {message.get('type')}")

    async def connect(self):
        """Connect to the agent server via WebSocket.

        This method:
        1. Establishes WebSocket connection with custom headers
        2. Adds agent_id and cap-related tags to identify this as a CapWorker client
        """
        agent_url = f"ws://{self.config.args.agent_host}:{self.config.args.agent_port}"
        logger.info(f"Connecting to agent server at {agent_url}")

        try:
            # Prepare headers to identify this as a CapWorker client
            headers = {
                "agent_id": self.config.args.agent_id,
                "client_type": "cap-worker",
                "cap_tag": "capx-robot-control",
            }

            # Connect to WebSocket server with headers
            self.websocket = await ws_connect(agent_url, additional_headers=headers)
            logger.info("✓ WebSocket connection established")
            logger.info(f"✓ Connected to agent server as {self.config.args.agent_id}")

        except Exception as e:
            logger.error(f"Failed to connect to agent server: {e}")
            raise

    async def disconnect(self):
        """Disconnect from the agent server."""

        if self.websocket:
            await self.websocket.close()
            self.websocket = None
            logger.info("Disconnected from agent server")

    def run_trial(
        self, task_goal: str, trial: int = 0, multi_turn_prompt: str | None = None
    ):
        """Execute a single trial by communicating with the agent.

        This method implements the core logic from _run_single_trial but
        communicates with the agent via WebSocket at each step.

        Args:
            trial: Trial number
            multi_turn_prompt: Optional multi-turn prompt template
        """
        from capx.envs.runner import _run_trial_with_retries

        self.env.change_goal(task_goal)
        return _run_trial_with_retries(
            self.env, trial, self.args, self.worker_config, multi_turn_prompt
        )

    async def start(self):
        """Start the CapWorker by connecting to agent and listening for tasks.

        This method:
        1. Connects to the agent server
        2. Enters a loop waiting for WebSocket messages
        3. When receiving 'cap_task' message, executes run_trial
        4. Continues listening for more tasks
        """
        agent_url = f"ws://{self.config.args.agent_host}:{self.config.args.agent_port}"
        logger.info(f"Starting CapWorker, connecting to: {agent_url}")

        # Create environment instance
        if self.env_factory is None:
            raise RuntimeError("Environment not initialized. Provide config_path.")
        self.env = instantiate(self.env_factory)
        # parse output dir
        if self.worker_config["output_dir"]:
            parts = self.worker_config["output_dir"].split("/")
            parts.insert(-1, str(self.args.model).replace("/", "_"))
            new_out_dir = "/".join(parts)
            Path(new_out_dir).mkdir(parents=True, exist_ok=True)
            self.worker_config["output_dir"] = new_out_dir

        # Connect to agent server
        await self.connect()

        # Send extern tools
        agent_id = self.config.args.agent_id
        extern_tools = [
            {
                "name": "trigger_cap_task",
                "description": f"Send a task instruction to Robot {agent_id} for execution. This tool should ONLY be called when you need to send a task to the {agent_id} for robot control or environment interaction. Do not call this tool for general conversation or information queries. The task will be executed by the CapWorker and results will be returned.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "The task instruction to be executed by the robot. Describe clearly what the robot should do, including objects to manipulate, actions to perform, and any specific requirements.",
                        }
                    },
                    "required": ["task"],
                },
                "mockResponse": {
                    "success": True,
                    "message": "Task sent to CapWorker successfully",
                },
            }
        ]
        await self._send_message({"type": "extern_tools", "tools": extern_tools})

        try:
            # Main loop: wait for tasks from agent
            logger.info("CapWorker ready, waiting for tasks...")

            while True:
                try:
                    # Get message from queue (with expected type "cap_task" or other control messages)
                    data = await self.websocket.recv()
                    message = json.loads(data)
                    msg_type = message.get("type", "unknown")
                    logger.info(f"Received message type: {msg_type}")

                    if msg_type == "cap_task":
                        try:
                            args = message.get("args", {})
                            for k, v in args.items():
                                setattr(self.args, k, v)
                            result = self.run_trial(
                                message["content"],
                                multi_turn_prompt=message.get("multi_turn_prompt"),
                            )

                            # Send result back to agent
                            await self._send_message(
                                {
                                    "type": "task_result",
                                    "result": {
                                        "success": result.get("success", False),
                                        "reward": result.get("reward", 0.0),
                                        "sandbox_rc": result.get("sandbox_rc", -1),
                                        "task_completed": result.get(
                                            "task_completed", False
                                        ),
                                        "code_path": result.get("code_path", ""),
                                        "num_regenerations": result.get(
                                            "num_regenerations", 0
                                        ),
                                        "num_finishes": result.get("num_finishes", 0),
                                        "num_code_blocks": result.get(
                                            "num_code_blocks", 0
                                        ),
                                    },
                                }
                            )

                        except Exception as e:
                            import traceback

                            traceback.print_exc()
                            await self._send_message(
                                {
                                    "type": "task_result",
                                    "result": {"success": False, "error": str(e)},
                                }
                            )

                    elif msg_type == "shutdown":
                        # Graceful shutdown request
                        logger.info("Received shutdown signal")
                        break
                    elif msg_type == "ping":
                        # Respond to ping
                        await self._send_message({"type": "pong"})
                    else:
                        logger.warning(f"Unknown message type: {msg_type}")

                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    import traceback

                    traceback.print_exc()

        finally:
            # Disconnect from agent server
            await self.disconnect()
            logger.info("CapWorker stopped")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(args: CapWorkerArgs) -> None:
    """Load config and start CapWorker."""
    from capx.utils.launch_utils import _load_config

    # Load environment configuration
    env_factory, config, _ = _load_config(args)
    if config.get("model"):
        args.model = config["model"]
    if config.get("visual_differencing_model"):
        args.visual_differencing_model = config["visual_differencing_model"]

    # Create CapWorkerConfig
    worker_config = CapWorkerConfig(args=args, env_factory=env_factory, config=config)

    # Create and start CapWorker
    worker = CapWorker(worker_config)

    try:
        asyncio.run(worker.start())
    except KeyboardInterrupt:
        logger.info("Shutting down CapWorker...")
    except Exception as e:
        logger.error(f"CapWorker failed: {e}")
        raise


if __name__ == "__main__":
    main(tyro.cli(CapWorkerArgs))
