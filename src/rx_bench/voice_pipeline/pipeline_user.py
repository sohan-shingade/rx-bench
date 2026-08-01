"""A tau2 user whose outgoing turns pass through a real speech channel.

``UserSimulator`` puts an LLM in the caller's seat. :class:`PipelineUser` keeps
that LLM but forces every turn it produces through TTS -> channel degradation
-> Deepgram ASR before the agent sees it. The agent under test therefore
receives **the ASR transcript, not the script** — exactly what a production
ASR->LLM pipeline receives — while everything else (orchestrator, ``llm_agent``
scaffold, tools, policy, environment, evaluator) is stock tau2. This is the
same integration pattern as ``rx_bench.live.human_user``: substitute only the
producer of user turns, change nothing else, and the run comes out as an
ordinary ``results.json`` that every existing scorer reads.

Transcript-injection convention (tau-voice compliant)
-----------------------------------------------------
* The message appended to the *shared* conversation — what the agent's context,
  the saved ``results.json``, and every scorer see — carries the ASR
  transcript.
* The user simulator's *own* state keeps the original text: a caller knows
  what they said, not what the phone line did to it. Without this the user-sim
  would start "remembering" the garbled version of its own speech and drift.
* Control sentinels (``###STOP###``, ``###TRANSFER###``, ``###OUT-OF-SCOPE###``)
  and tool-call turns bypass the channel entirely: they are orchestration
  signals, not speech, and mangling them would break termination.
* The agent's replies pass through as text (transcript injection on the agent
  ear only). The agent's own TTS leg is acknowledged but not audibly rendered
  in v1 — see README.md, "Limitations".

Every pipelined turn is logged twice: a compact summary embedded in the
``UserMessage.raw_data`` (travels inside ``results.json``) and a full record
appended to ``pipeline_turns.jsonl`` in the run directory (original text, WAV
paths, WER, word confidences, timings).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Optional

from tau2.data_model.message import UserMessage
from tau2.user.user_simulator import UserSimulator
from tau2.user.user_simulator_base import (
    OUT_OF_SCOPE,
    STOP,
    TRANSFER,
    UserState,
    ValidUserInputMessage,
)

from .pipeline import SpeechChannel, TurnRecord

SENTINELS = (STOP, TRANSFER, OUT_OF_SCOPE)


class PipelineUser(UserSimulator):
    """LLM user simulator whose speech reaches the agent through a phone line.

    Class-level wiring (channel, turn log, task-id lookup) is installed by
    :func:`bind_pipeline_user`, because tau2's registry constructs the user
    itself with a fixed keyword set (see ``runner/build.py:build_user``) and
    offers no place to pass extra collaborators — the same constraint that
    shaped ``rx_bench.live.talk.bind_channel``.
    """

    channel: SpeechChannel  # installed by bind_pipeline_user
    turn_log_path: Optional[Path] = None
    task_id_by_instructions: dict[str, str] = {}
    _log_lock = threading.Lock()

    def __init__(self, **kwargs: object):
        super().__init__(**kwargs)  # tools, instructions, llm, llm_args, persona_config
        self._turn_index = 0
        self._conversation_key: Optional[str] = None

    # -- identity ----------------------------------------------------------

    def _conversation(self) -> str:
        """Stable key for this conversation: <task-id>-t<trial-seed>.

        The registry hands the user only its instructions, so the runner
        provides an instructions->task-id map. The llm seed (set per trial by
        the orchestrator via ``set_seed``) distinguishes trials.
        """
        if self._conversation_key is None:
            task_id = self.task_id_by_instructions.get(
                str(self.instructions), "unknown-task"
            )
            seed = self.llm_args.get("seed", "x")
            self._conversation_key = f"{task_id}-s{seed}"
        return self._conversation_key

    # -- the seam ----------------------------------------------------------

    @staticmethod
    def _is_speech(message: UserMessage) -> bool:
        if message.is_tool_call() or not message.content:
            return False
        return not any(s in message.content for s in SENTINELS)

    def generate_next_message(
        self, message: ValidUserInputMessage, state: UserState
    ) -> tuple[UserMessage, UserState]:
        # The base method appends the agent turn to the user's private state
        # and asks the LLM for the caller's next line (original, clean text).
        original = self._generate_next_message(message, state)

        delivered = original
        if self._is_speech(original):
            record = self.channel.process(
                original.content, self._conversation(), self._turn_index
            )
            self._log(record)
            transcript = record.transcript.strip() or "[inaudible]"
            delivered = original.model_copy(deep=True)
            delivered.content = transcript
            raw = dict(delivered.raw_data or {})
            raw["voice_pipeline"] = record.summary()
            delivered.raw_data = raw
        self._turn_index += 1

        # Private memory keeps what the caller SAID; the wire carries what the
        # ASR HEARD. This asymmetry is the entire point of the agent class.
        state.messages.append(original)
        return delivered, state

    # -- logging -----------------------------------------------------------

    def _log(self, record: TurnRecord) -> None:
        if self.turn_log_path is None:
            return
        line = json.dumps(record.as_dict(), ensure_ascii=False)
        with self._log_lock:
            self.turn_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.turn_log_path, "a", encoding="utf-8") as fp:
                fp.write(line + "\n")


def bind_pipeline_user(
    channel: SpeechChannel,
    turn_log_path: Optional[Path],
    task_id_by_instructions: dict[str, str],
) -> type:
    """Produce a registrable user class with its collaborators closed over."""

    class BoundPipelineUser(PipelineUser):
        pass

    BoundPipelineUser.channel = channel
    BoundPipelineUser.turn_log_path = turn_log_path
    BoundPipelineUser.task_id_by_instructions = dict(task_id_by_instructions)
    BoundPipelineUser.__name__ = "BoundPipelineUser"
    return BoundPipelineUser
