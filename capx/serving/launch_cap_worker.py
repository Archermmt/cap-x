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

import os
import shutil
import asyncio
import json
import logging
from dataclasses import dataclass, asdict
from typing import Any

import tyro
from pathlib import Path
from websockets.asyncio.client import connect as ws_connect

from capx.envs.launch import LaunchArgs
from capx.envs.configs.instantiate import instantiate
from capx.envs.configs.loader import DictLoader

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

    agent_id: str = "cap"
    """Agent ID to identify this worker to the server."""

    robot_name: str = "jarvis"
    """Name of the robot being controlled."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class CapWorker:
    """WebSocket client for agent interaction with CaP-X environment.

    This worker connects to an external agent server via WebSocket and
    executes the complete trial flow, sending queries to the agent and
    receiving responses for code generation and decision making.
    """

    def __init__(self, args: CapWorkerArgs):
        """Initialize CapWorker with command-line arguments.

        This method loads environment configuration and initializes all components.

        Args:
            args: Command-line arguments containing configuration path and other settings
        """
        from capx.utils.launch_utils import _load_config

        # Load environment configuration
        self.args = args
        self.env_factory, self.config, _ = _load_config(self.args)
        config_path = os.path.expanduser(args.config_path)
        configs_dict = DictLoader.load([config_path])
        for key in ["agent_host", "agent_port", "http_port", "agent_id", "robot_name"]:
            if key in configs_dict:
                setattr(self.args, key, configs_dict[key])

        self.websocket = None
        self.env = None
        agent_url = f"ws://{self.args.agent_host}:{self.args.agent_port}"
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
        agent_url = f"ws://{self.args.agent_host}:{self.args.agent_port}"
        logger.info(f"Connecting to agent server at {agent_url}")

        try:
            # Prepare headers to identify this as a CapWorker client
            headers = {
                "agent_id": self.args.agent_id,
                "client_type": "cap-worker",
                "cap_tag": "capx-robot-control",
            }

            # Connect to WebSocket server with headers
            self.websocket = await ws_connect(
                agent_url,
                additional_headers=headers,
                ping_interval=120,
                ping_timeout=300,
            )
            logger.info("✓ WebSocket connection established")
            logger.info(f"✓ Connected to agent server as {self.args.agent_id}")

        except Exception as e:
            logger.error(f"Failed to connect to agent server: {e}")
            raise

    async def disconnect(self):
        """Disconnect from the agent server."""

        if self.websocket:
            await self.websocket.close()
            self.websocket = None
            logger.info("Disconnected from agent server")

    def _encode_video_to_base64(self, video_path: str) -> str | None:
        """Encode a video file to base64 string.

        Args:
            video_path: Path to the video file

        Returns:
            Base64 encoded string or None if encoding fails
        """
        import base64

        try:
            with open(video_path, "rb") as f:
                video_data = f.read()
                base64_video = base64.b64encode(video_data).decode("utf-8")
                return f"data:video/mp4;base64,{base64_video}"
        except Exception as e:
            logger.error(f"Failed to encode video {video_path}: {e}")
            return None

    def _get_task_records(self, result, send_all=False):
        """Send task record message with encoded video if available.

        Args:
            result: TrialSummary object containing code_path and other info
        """
        if not result.code_path:
            logger.warning("No code_path in result, skipping task_record")
            return

        trial_dir = Path(result.code_path).parent
        video_files = list(trial_dir.glob("*.mp4"))
        video_files.sort(key=lambda f: f.stat().st_size, reverse=True)

        if not video_files:
            logger.info("No video files found in trial directory")
            return

        # Encode all video files
        records = []
        for video_file in video_files:
            logger.info(f"Encoding video: {video_file.name}")
            base64_video = self._encode_video_to_base64(str(video_file))
            if base64_video:
                records.append(base64_video)
        if not send_all:
            records = records[:1]
        return records

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
            self.env, trial, self.args, self.config, multi_turn_prompt
        )

    async def start(self):
        """Start the CapWorker by connecting to agent and listening for tasks.

        This method:
        1. Connects to the agent server
        2. Enters a loop waiting for WebSocket messages
        3. When receiving 'cap_task' message, executes run_trial
        4. Continues listening for more tasks
        """
        agent_url = f"ws://{self.args.agent_host}:{self.args.agent_port}"
        logger.info(f"Starting CapWorker, connecting to: {agent_url}")

        # Create environment instance and connect
        if self.env_factory is None:
            raise RuntimeError("Environment not initialized. Provide config_path.")
        self.env = instantiate(self.env_factory)
        await self.connect()

        # Send extern tools
        robot_name = self.args.robot_name
        extern_tools = [
            {
                "name": "trigger_cap_task",
                "description": f"Send a task instruction to Robot '{robot_name}' for execution. This tool should ONLY be called when you need to send a task to the '{robot_name}' for robot control or environment interaction. Do not call this tool for general conversation or information queries. The task will be executed by the CapWorker and results will be returned.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "The task instruction to be executed by the robot. Describe clearly what the robot should do, including objects to manipulate, actions to perform, and any specific requirements. IMPORTANT: This field value must be entirely in English.",
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

        # Main loop: wait for tasks from agent
        logger.info("CapWorker ready, waiting for tasks...")
        while True:
            # Get message from queue (with expected type "cap_task" or other control messages)
            data = await self.websocket.recv()
            message = json.loads(data)
            msg_type = message.get("type", "unknown")
            logger.info(f"Received message type: {msg_type}")

            if msg_type == "cap_task":
                try:
                    # Clear output directory before running trial
                    if self.config.get("output_dir"):
                        output_path = Path(self.config["output_dir"])
                        if output_path.exists():
                            shutil.rmtree(output_path)
                        output_path.mkdir(parents=True, exist_ok=True)

                    args = message.get("args", {})
                    for k, v in args.items():
                        setattr(self.args, k, v)
                    result = self.run_trial(
                        message["content"],
                        multi_turn_prompt=message.get("multi_turn_prompt"),
                    )

                    # Send task record with video before sending result
                    records = self._get_task_records(result)
                    if records:
                        logger.info(f"Sending task_record with {len(records)} video(s)")
                        await self._send_message(
                            {"type": "task_record", "records": records}
                        )
                    # Send result back to agent
                    await self._send_message(
                        {"type": "task_result", "result": asdict(result)}
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
            else:
                logger.warning(f"Unknown message type: {msg_type}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(args: CapWorkerArgs) -> None:
    """Start CapWorker."""
    # Create and start CapWorker
    worker = CapWorker(args)

    try:
        asyncio.run(worker.start())
    except KeyboardInterrupt:
        logger.info("Shutting down CapWorker...")
    except Exception as e:
        logger.error(f"CapWorker failed: {e}")
        raise


if __name__ == "__main__":
    main(tyro.cli(CapWorkerArgs))
