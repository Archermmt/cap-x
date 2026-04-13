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
import copy
import json
import logging
import os
import signal
from dataclasses import dataclass
from typing import Any

import tyro
import websockets
from websockets.asyncio.client import connect as ws_connect

from capx.envs.launch import LaunchArgs
from capx.utils.launch_utils import (
    _build_multi_turn_decision_prompt,
    _build_multi_turn_decision_prompt_legacy,
    _extract_code,
    _get_visual_feedback,
    _parse_multi_turn_decision,
)

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
    agent_url: str = "ws://localhost:8765/agent"
    """WebSocket URL of the agent server to connect to."""

    auth_token: str | None = None
    """Authentication token for WebSocket connection (optional)."""

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

    config_dict: dict[str, Any] = None
    """Configuration dictionary loaded from YAML."""

    def __post_init__(self):
        if self.config_dict is None:
            self.config_dict = {}


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
        self.config_dict = config.config_dict
        self.websocket = None
        self.env = None

        logger.info(f"CapWorker initialized, will connect to: {config.args.agent_url}")

    async def _send_message(self, message: dict[str, Any]):
        """Send a message to the agent via WebSocket.

        Args:
            message: Message dictionary to send
        """
        if self.websocket is None:
            raise RuntimeError("Not connected to agent server")

        await self.websocket.send(json.dumps(message))
        logger.debug(f"Sent to agent: {message.get('type')}")

    async def _receive_message(self) -> dict[str, Any]:
        """Receive a message from the agent via WebSocket.

        Returns:
            Received message as dictionary
        """
        if self.websocket is None:
            raise RuntimeError("Not connected to agent server")

        data = await self.websocket.recv()
        message = json.loads(data)
        logger.debug(f"Received from agent: {message.get('type')}")
        return message

    async def connect(self):
        """Connect to the agent server via WebSocket with protocol headers.

        This method:
        1. Establishes WebSocket connection
        2. Sends authentication message if configured (per CapProto protocol)
        3. Waits for connection acceptance
        """
        logger.info(f"Connecting to agent server at {self.config.args.agent_url}")

        try:
            # Add agent_id to URL query parameters if not present
            agent_url = self.config.args.agent_url
            if "agent_id=" not in agent_url:
                separator = "&" if "?" in agent_url else "?"
                agent_url = (
                    f"{agent_url}{separator}agent_id={self.config.args.agent_id}"
                )

            # Connect to WebSocket server
            self.websocket = await ws_connect(agent_url)
            logger.info("✓ WebSocket connection established")

            # Send authentication message if token is configured (per CapProto protocol)
            if self.config.args.auth_token:
                auth_message = {
                    "type": "auth",
                    "token": self.config.args.auth_token,
                    "agent_id": self.config.args.agent_id,
                }
                await self.websocket.send(json.dumps(auth_message))
                logger.debug("Sent authentication message")

                # Wait for authentication response
                try:
                    import asyncio

                    response_data = await asyncio.wait_for(
                        self.websocket.recv(), timeout=10.0
                    )
                    response = json.loads(response_data)

                    if response.get("type") == "auth_response":
                        if response.get("status") == "accepted":
                            logger.info("✓ Authentication successful")
                        else:
                            logger.error("✗ Authentication failed")
                            await self.websocket.close()
                            self.websocket = None
                            raise RuntimeError("Authentication rejected by server")
                    else:
                        logger.warning(
                            f"Unexpected response type: {response.get('type')}"
                        )

                except asyncio.TimeoutError:
                    logger.warning("Authentication response timeout, continuing anyway")
                except json.JSONDecodeError:
                    logger.warning("Invalid authentication response format")

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

    async def run_trial(self, trial: int = 0, multi_turn_prompt: str | None = None):
        """Execute a single trial by communicating with the agent.

        This method implements the core logic from _run_single_trial but
        communicates with the agent via WebSocket at each step.

        Args:
            trial: Trial number
            multi_turn_prompt: Optional multi-turn prompt template
        """
        try:
            logger.info(f"Starting trial {trial}")

            # Reset environment
            obs, _ = self.env.reset(options={"trial": trial}, seed=trial)

            # Reset SIGALRM timer
            remaining = signal.alarm(0)
            if remaining > 0:
                signal.alarm(1000)

            obs["full_prompt"] = copy.deepcopy(obs["full_prompt"])

            # Enable video capture if configured
            use_wrist = self.config_dict.get("use_wrist_camera", False)
            if self.config_dict.get("record_video") and hasattr(
                self.env, "enable_video_capture"
            ):
                self.env.enable_video_capture(True, clear=True, wrist_camera=use_wrist)

            # Shared trial state
            code_blocks = []
            code_block_metadata = []
            all_responses = []
            stderr_history = []
            num_regenerations = 0
            num_finishes = 0
            info_step = {"sandbox_rc": -1, "stdout": "", "stderr": ""}
            reward = 0.0
            terminated = truncated = False

            # Capture initial visual feedback
            logger.info("Capturing initial visual feedback")
            visual_feedback_imgs, visual_feedback_base64_history, task_description = (
                self._capture_initial_visual_feedback(obs)
            )

            # Initial code generation
            logger.info("Requesting initial code from agent")
            if self.config_dict.get("use_oracle_code"):
                raw_code = self.env.oracle_code
                reasoning = None
            else:
                # Query agent for initial code
                await self._send_message(
                    {
                        "type": "query_code",
                        "prompt": obs["full_prompt"],
                        "task_description": task_description,
                    }
                )

                # Wait for agent response
                response = await self._receive_message()

                if response.get("type") != "code_response":
                    raise ValueError(
                        f"Expected code_response, got {response.get('type')}"
                    )

                raw_code = response.get("content", "")
                reasoning = response.get("reasoning")

            # Parse initial code into blocks
            initial_blocks = _extract_code(raw_code)
            code_blocks.extend(initial_blocks)
            code_block_metadata.extend(
                [{"generation": 0, "regenerated": False}] * len(initial_blocks)
            )

            all_responses.append(
                {
                    "block_idx": [0],
                    "code_blocks": initial_blocks,
                    "decision": "initial",
                    "initial_prompt": copy.deepcopy(obs["full_prompt"]),
                    "reasoning": reasoning or "",
                }
            )

            # Execute code blocks
            code_block_idx = 0
            MULTITURN_LIMIT = 10

            while (
                code_block_idx < len(code_blocks) and code_block_idx <= MULTITURN_LIMIT
            ):
                code = code_blocks[code_block_idx]
                code_block_idx += 1

                # Execute code block
                logger.info(f"Executing code block {code_block_idx}/{len(code_blocks)}")
                obs_next, reward, terminated, truncated, info_step = self.env.step(code)
                obs = obs_next

                # Check if we need multi-turn decision
                if multi_turn_prompt:
                    if "terminated episode" in info_step["stderr"]:
                        truncated = True
                        break

                    # Handle multi-turn decision
                    logger.info("Requesting multi-turn decision from agent")
                    decision, new_code = await self._handle_multi_turn_step(
                        obs,
                        multi_turn_prompt,
                        code_blocks,
                        code_block_idx,
                        info_step,
                        task_description,
                        visual_feedback_base64_history,
                        stderr_history,
                    )

                    if decision == "regenerate":
                        logger.info("Agent chose to regenerate code")
                        new_blocks = _extract_code(new_code)
                        all_responses.append(
                            {
                                "block_idx": [code_block_idx],
                                "code_blocks": new_blocks,
                                "decision": "regenerate",
                                "reasoning": "",
                            }
                        )
                        del code_blocks[code_block_idx:]
                        del code_block_metadata[code_block_idx:]
                        code_blocks.extend(new_blocks)
                        code_block_metadata.extend(
                            [
                                {
                                    "generation": num_regenerations + 1,
                                    "regenerated": True,
                                    "regenerated_at_idx": code_block_idx,
                                }
                            ]
                            * len(new_blocks)
                        )
                        num_regenerations += 1

                    elif decision == "finish":
                        all_responses.append(
                            {
                                "decision": "finish",
                                "reasoning": "",
                            }
                        )
                        logger.info("Agent chose to finish")
                        num_finishes += 1
                        break

                logger.info(f"Code block {code_block_idx} completed")

                # Save intermediate artifacts
                final_code = self._annotate_code_blocks(
                    code_blocks, code_block_metadata
                )
                self._save_trial_artifacts(
                    trial,
                    info_step["sandbox_rc"],
                    reward,
                    info_step.get("task_completed", False),
                    final_code,
                    raw_code,
                    all_responses,
                    ["-" * 100, "Generated program:", final_code],
                    visual_feedback_imgs,
                )

            logger.info("All code blocks executed")

            # Build final summary
            final_code = self._annotate_code_blocks(code_blocks, code_block_metadata)
            num_code_blocks = len(code_blocks)

            # Override sandbox_rc for terminated-episode stderr
            if "executing action in terminated episode" in info_step["stderr"]:
                info_step["sandbox_rc"] = 0

            stderr = (
                "\n\n".join(stderr_history) if stderr_history else info_step["stderr"]
            )
            log_lines = self._build_log_lines(
                final_code,
                info_step,
                reward,
                terminated,
                truncated,
                num_regenerations,
                num_finishes,
                num_code_blocks,
                stderr_override=stderr,
            )

            # Save final artifacts
            code_path = self._save_trial_artifacts(
                trial,
                info_step["sandbox_rc"],
                reward,
                info_step.get("task_completed", False),
                final_code,
                raw_code,
                all_responses,
                log_lines,
                visual_feedback_imgs,
            )

            success = info_step["sandbox_rc"] == 0

            logger.info(
                f"Trial {trial} completed successfully={success}, reward={reward:.3f}"
            )

            return {
                "trial": trial,
                "success": success,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "sandbox_rc": info_step["sandbox_rc"],
                "log": "\n".join(log_lines),
                "task_completed": info_step.get("task_completed", None),
                "code_path": code_path,
                "num_regenerations": num_regenerations,
                "num_finishes": num_finishes,
                "num_code_blocks": num_code_blocks,
            }

        except Exception as e:
            logger.error(f"Error in trial execution: {e}")
            import traceback

            traceback.print_exc()
            raise
        finally:
            # Clean up environment
            if hasattr(self, "env") and self.env is not None:
                try:
                    self.env.close()
                except:
                    pass

    async def _handle_multi_turn_step(
        self,
        obs: dict[str, Any],
        multi_turn_prompt: str,
        code_blocks: list[str],
        code_block_idx: int,
        info_step: dict[str, Any],
        task_description: str,
        visual_feedback_base64_history: list[str],
        stderr_history: list[str],
    ) -> tuple[str, str]:
        """Handle multi-turn decision step by querying the agent.

        Args:
            obs: Current observation
            multi_turn_prompt: Multi-turn prompt template
            code_blocks: List of executed code blocks
            code_block_idx: Current code block index
            info_step: Information about the last execution step
            task_description: Task description
            visual_feedback_base64_history: History of visual feedback
            stderr_history: History of stderr outputs

        Returns:
            Tuple of (decision, new_code) where decision is "regenerate", "finish", or "continue"
        """
        executed_code = "\n".join(code_blocks[:code_block_idx])
        complete_multi_turn_prompt = multi_turn_prompt.format(
            executed_code=executed_code,
            console_stdout=info_step["stdout"],
            console_stderr=info_step["stderr"],
        )

        if info_step["stderr"] != "":
            stderr_history.append(info_step["stderr"])

        # Capture visual feedback if applicable
        visual_feedback_base64 = None
        needs_visual = (self.config_dict.get("use_visual_feedback", False)) or (
            self.config_dict.get("use_img_differencing", False)
        )

        if needs_visual and hasattr(self.env, "render"):
            vf_base64, vf_img = _get_visual_feedback(self.env)
            visual_feedback_base64_history.append(vf_base64)
            visual_feedback_base64 = vf_base64

        # Determine differencing feedback
        differencing_feedback = None
        if (
            self.config_dict.get("use_img_differencing", False)
            and len(visual_feedback_base64_history) >= 2
        ):
            differencing_feedback = self._get_visual_differencing_feedback(
                task_description, visual_feedback_base64_history
            )

        # Only pass visual feedback to prompt if visual_feedback is enabled
        if not self.config_dict.get("use_visual_feedback", False):
            visual_feedback_base64 = None

        # Build decision prompt
        if self.args.use_legacy_multi_turn_decision_prompt:
            decision_prompt = _build_multi_turn_decision_prompt_legacy(
                obs,
                complete_multi_turn_prompt,
                visual_feedback_base64,
                differencing_feedback,
            )
        else:
            decision_prompt = _build_multi_turn_decision_prompt(
                obs,
                complete_multi_turn_prompt,
                visual_feedback_base64,
                differencing_feedback,
            )

        # Query agent for decision
        await self._send_message(
            {
                "type": "query_decision",
                "prompt": decision_prompt,
                "task_description": task_description,
            }
        )

        # Wait for agent response
        response = await self._receive_message()

        if response.get("type") != "decision_response":
            raise ValueError(f"Expected decision_response, got {response.get('type')}")

        content = response.get("content", "")
        decision, new_code = _parse_multi_turn_decision(content)

        return decision, new_code

    def _capture_initial_visual_feedback(
        self,
        obs: dict[str, Any],
    ) -> tuple[list, list[str], str]:
        """Capture the initial environment image and optionally describe it.

        Returns:
            (visual_feedback_imgs, visual_feedback_base64_history, task_description)
        """

        visual_feedback_imgs = []
        visual_feedback_base64_history = []
        task_description = ""

        needs_visual = (
            (self.config_dict.get("use_visual_feedback", False))
            or (self.config_dict.get("use_img_differencing", False))
            or self.config_dict.get("use_video_differencing", False)
        )

        if not (needs_visual and hasattr(self.env, "render")):
            return (
                visual_feedback_imgs,
                visual_feedback_base64_history,
                task_description,
            )

        initial_base64, initial_img = _get_visual_feedback(self.env)
        visual_feedback_imgs.append(initial_img)
        visual_feedback_base64_history.append(initial_base64)
        task_description = obs["full_prompt"][-1]["content"][0]["text"]

        # Append image to the prompt for VLM visual feedback
        if self.config_dict.get("use_visual_feedback", False):
            obs["full_prompt"][-1]["content"][0][
                "text"
            ] += "\n\nIncluded below is an image of the initial state of the environment."
            obs["full_prompt"][-1]["content"].append(
                {"type": "image_url", "image_url": {"url": initial_base64}}
            )

        return visual_feedback_imgs, visual_feedback_base64_history, task_description

    def _get_visual_differencing_feedback(
        self,
        task_description: str,
        visual_feedback_base64_history: list[str],
    ) -> str | None:
        """Query a VLM to describe what changed between the two most recent frames."""
        if len(visual_feedback_base64_history) < 2:
            return None

        # This would normally call a VLM, but for now we'll return None
        # In a real implementation, you'd call your visual differencing model here
        return None

    def _annotate_code_blocks(
        self,
        code_blocks: list[str],
        code_block_metadata: list[dict[str, Any]],
    ) -> str:
        """Join code blocks into a single string with ``# Code block N`` headers."""
        annotated = []
        for i, (block, metadata) in enumerate(zip(code_blocks, code_block_metadata)):
            annotated.append(f"# Code block {i}\n{block}")
        return "\n\n".join(annotated)

    def _build_log_lines(
        self,
        final_code: str,
        info_step: dict[str, Any],
        reward: float,
        terminated: bool,
        truncated: bool,
        num_regenerations: int,
        num_finishes: int,
        num_code_blocks: int,
        *,
        prefix: str = "",
        stderr_override: str | None = None,
    ) -> list[str]:
        """Build the standard log-line list used for both normal and timeout summaries."""
        stderr = (
            stderr_override
            if stderr_override is not None
            else info_step.get("stderr", "")
        )
        lines = ["-" * 100]
        if prefix:
            lines.append(prefix)
        lines.extend(
            [
                "Generated program:",
                final_code if final_code else "(no program available)",
                "\n\nEnvironment response:",
                f"  Sandbox failed: {info_step.get('sandbox_rc', 1)}",
                f"  Stdout: {info_step.get('stdout', '')}",
                f"  Stderr: {stderr}",
                f"  Reward: {reward}",
                f"  Task Completed: {info_step.get('task_completed', False)}",
                f"  Terminated: {terminated}, Truncated: {truncated}",
                f"  Num Regenerations: {num_regenerations}",
                f"  Num Finishes: {num_finishes}",
                f"  Num Code Blocks: {num_code_blocks}",
                "-" * 100,
            ]
        )
        return lines

    def _save_trial_artifacts(
        self,
        trial: int,
        sandbox_rc: int,
        reward: float,
        task_completed: bool,
        final_code: str,
        raw_code: str,
        all_responses: list[dict[str, Any]],
        log_lines: list[str],
        visual_feedback_imgs: list,
        ensemble_data: dict | None = None,
        multiturn_ensemble_data: list[dict[str, Any]] | None = None,
    ) -> str:
        """Save trial artifacts including code, logs, and images."""
        if not self.config_dict.get("output_dir"):
            return ""

        # Create trial directory
        trial_dir = os.path.join(
            self.config_dict["output_dir"],
            f"trial_{trial:02d}_sandboxrc_{sandbox_rc}_reward_{reward:.3f}"
            f"_taskcompleted_{int(task_completed)}",
        )
        os.makedirs(trial_dir, exist_ok=True)

        # Save code
        code_path = os.path.join(trial_dir, "generated_code.py")
        with open(code_path, "w") as f:
            f.write(final_code)

        # Save raw code
        raw_code_path = os.path.join(trial_dir, "raw_code.txt")
        with open(raw_code_path, "w") as f:
            f.write(raw_code)

        # Save responses
        responses_path = os.path.join(trial_dir, "all_responses.json")
        with open(responses_path, "w") as f:
            json.dump(all_responses, f, indent=2)

        # Save log
        log_path = os.path.join(trial_dir, "trial_log.txt")
        with open(log_path, "w") as f:
            f.write("\n".join(log_lines))

        # Save visual feedback images
        for i, img in enumerate(visual_feedback_imgs):
            if img is not None:
                img_path = os.path.join(trial_dir, f"visual_feedback_{i:02d}.png")
                img.save(img_path)

        return code_path

    async def start(self):
        """Start the CapWorker by connecting to agent and listening for tasks.

        This method:
        1. Connects to the agent server
        2. Enters a loop waiting for WebSocket messages
        3. When receiving 'cap_task' message, executes run_trial
        4. Continues listening for more tasks
        """
        logger.info(f"Starting CapWorker, connecting to: {self.config.args.agent_url}")

        # Create environment instance
        if self.env_factory is None:
            raise RuntimeError("Environment not initialized. Provide config_path.")
        self.env = self.env_factory()

        # Connect to agent server
        await self.connect()

        try:
            # Main loop: wait for tasks from agent
            logger.info("CapWorker ready, waiting for tasks...")
            task_count = 0

            while True:
                try:
                    # Wait for incoming message
                    message_data = await self.websocket.recv()
                    message = json.loads(message_data)
                    msg_type = message.get("type", "")

                    logger.info(f"Received message type: {msg_type}")

                    if msg_type == "cap_task":
                        # Execute trial when receiving cap_task message
                        task_count += 1
                        trial_num = message.get("trial", task_count - 1)
                        multi_turn_prompt = message.get("multi_turn_prompt")

                        logger.info(f"Starting trial {trial_num} (task #{task_count})")

                        try:
                            result = await self.run_trial(
                                trial=trial_num,
                                multi_turn_prompt=multi_turn_prompt,
                            )

                            logger.info(f"Trial {trial_num} completed: {result}")

                            # Send result back to agent
                            await self._send_message(
                                {
                                    "type": "task_result",
                                    "trial": trial_num,
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
                                    "num_code_blocks": result.get("num_code_blocks", 0),
                                }
                            )

                        except Exception as e:
                            logger.error(f"Trial {trial_num} failed: {e}")
                            import traceback

                            traceback.print_exc()

                            # Send error result to agent
                            await self._send_message(
                                {
                                    "type": "task_result",
                                    "trial": trial_num,
                                    "success": False,
                                    "error": str(e),
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

                except websockets.exceptions.ConnectionClosed:
                    logger.warning("WebSocket connection closed")
                    break
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse message: {e}")
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
    env_factory, config_dict, _ = _load_config(args)

    # Create CapWorkerConfig
    worker_config = CapWorkerConfig(
        args=args,
        env_factory=env_factory,
        config_dict=config_dict,
    )

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
