"""A tau2 user that is an actual person.

``UserSimulator`` puts an LLM in the caller's seat and hands it a persona plus a
task. :class:`HumanUser` puts *you* there instead. Everything else in the
simulation is untouched --- same agent, same nineteen tools, same ``policy.md``,
same environment, same flight recorder --- so a conversation held here is
recorded in exactly the format the scorers already read.

The class deliberately implements nothing clever. Its whole job is to render an
agent turn onto a :class:`~rx_bench.live.channels.Channel` and wrap whatever comes
back into a ``UserMessage``. Any intelligence in the conversation is yours.
"""

from __future__ import annotations

import json
from typing import Optional

from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool
from tau2.user.user_simulator_base import (
    STOP,
    HalfDuplexUser,
    UserState,
    ValidUserInputMessage,
)

from .channels import Channel, TextChannel

#: Typed by the person to end the call. ``STOP`` is tau2's own sentinel: the
#: orchestrator watches for it, so hanging up here terminates the simulation the
#: same way a simulated caller's goodbye does, and the run is saved rather than
#: abandoned.
HANGUP_WORDS = {"/stop", "/quit", "/bye", "/hangup", "/exit"}


def render_agent_turn(message: ValidUserInputMessage, channel: Channel) -> None:
    """Put one agent-side message on the channel.

    Tool calls and their results are ``note``d rather than ``say``d: a person on
    a phone call does not hear ``find_patient(...)``, and a voice channel that
    read them aloud would be demonstrating something that never happens on a
    real line. They stay visible on screen because watching the tool calls land
    next to the words is most of the value of talking to it at all.
    """
    if isinstance(message, MultiToolMessage):
        for part in message.tool_messages:
            render_agent_turn(part, channel)
        return

    if isinstance(message, ToolMessage):
        body = (message.content or "").strip().replace("\n", " ")
        if len(body) > 240:
            body = body[:237] + "…"
        channel.note(f"↩ {body}" if body else "↩ (no content)")
        return

    if isinstance(message, AssistantMessage):
        if message.is_tool_call():
            for call in message.tool_calls or []:
                channel.note(f"⚙ {call.name}({_short_args(call.arguments)})")
        if message.content:
            spoken, envelope = unwrap_envelope(message.content)
            if envelope:
                # Announced, not hidden. The agent is supposed to answer a
                # caller in prose; a JSON wrapper is a protocol deviation, and
                # this project is precisely about deviations that get smoothed
                # over on the way to a human. What is stored on the run is
                # still the raw content -- only the delivery is unwrapped, so a
                # voice channel does not read braces aloud.
                channel.note(f"⚠ agent replied inside a {envelope} envelope")
            channel.say(spoken)
        return

    content = getattr(message, "content", None)
    if content:
        channel.say(str(content))


def unwrap_envelope(content: str) -> tuple[str, Optional[str]]:
    """Return ``(text_to_deliver, envelope_description)``.

    Some models wrap an otherwise ordinary reply in ``{"message": "..."}``.
    Delivering that verbatim to a speech synthesiser is useless, and delivering
    it silently unwrapped would conceal that it happened, so the caller gets the
    text and the operator gets told.
    """
    stripped = content.strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return content, None
    try:
        payload = json.loads(stripped)
    except (ValueError, TypeError):
        return content, None
    if not isinstance(payload, dict):
        return content, None
    for key in ("message", "response", "text", "reply", "content"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            extra = sorted(k for k in payload if k != key)
            described = f"json {{{key}}}"
            if extra:
                described += " with " + ", ".join(extra)
            return value, described
    return content, None


def _short_args(arguments: Optional[dict]) -> str:
    if not arguments:
        return ""
    parts = []
    for key, value in arguments.items():
        text = str(value)
        if len(text) > 40:
            text = text[:37] + "…"
        parts.append(f"{key}={text}")
    return ", ".join(parts)


class HumanUser(HalfDuplexUser[UserState]):
    """The caller is a person at a keyboard or a microphone."""

    #: The most recently constructed instance. tau2's registry builds the user
    #: itself and hands the object to the orchestrator, so a caller that wants
    #: the live transcript after a crash has no other handle on it.
    latest: "Optional[HumanUser]" = None

    def __init__(
        self,
        tools: Optional[list[Tool]] = None,
        instructions: Optional[str] = None,
        channel: Optional[Channel] = None,
        # Accepted and ignored: tau2's build_user passes these to every
        # registered user. A human has no model and no sampling arguments, and
        # raising on them would just make this class unusable through the
        # normal construction path.
        llm: Optional[str] = None,
        llm_args: Optional[dict] = None,
        **_ignored: object,
    ):
        super().__init__(instructions=instructions, tools=tools)
        self.channel = channel or TextChannel()
        self.turns = 0
        # Every message that crossed this seam, in order. tau2 assembles its
        # SimulationRun only after the orchestrator loop returns, so an
        # exception anywhere in the loop throws the whole conversation away.
        # A person cannot say those words again; keeping a running copy is the
        # difference between a lost call and a partial one.
        self.transcript: list[Message] = []
        HumanUser.latest = self

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> UserState:
        return UserState(
            system_messages=[
                SystemMessage(role="system", content="Human caller (live session).")
            ],
            messages=list(message_history or []),
        )

    @classmethod
    def is_stop(cls, message: UserMessage) -> bool:
        if message.is_tool_call() or message.content is None:
            return False
        return STOP in message.content

    def set_seed(self, seed: int) -> None:
        """No-op. Seeding a person is not a thing."""

    def generate_next_message(
        self, message: ValidUserInputMessage, state: UserState
    ) -> tuple[UserMessage, UserState]:
        self._capture(message)
        render_agent_turn(message, self.channel)

        spoken = self.channel.listen()
        if not spoken or spoken.strip().lower() in HANGUP_WORDS:
            # An empty read is EOF or ctrl-c. Ending with tau2's stop sentinel
            # rather than an exception means the partial conversation is still
            # finalised, evaluated and written out -- hanging up mid-call is a
            # normal way for a phone call to end, not a crash.
            content = f"{STOP} (caller hung up)"
        else:
            content = spoken
            self.turns += 1

        # cost=0.0 is the truth, not a placeholder: a human turn spends no
        # model tokens. Leaving it None makes the orchestrator's cost rollup
        # warn on every message it cannot price.
        user_message = UserMessage(role="user", content=content, cost=0.0)
        state.messages.append(user_message)
        self.transcript.append(user_message)
        return user_message, state

    def _capture(self, message: ValidUserInputMessage) -> None:
        """Record an agent-side message, flattening tool batches."""
        if isinstance(message, MultiToolMessage):
            self.transcript.extend(message.tool_messages)
        else:
            self.transcript.append(message)

    def stop(self, message=None, state=None) -> None:
        self.channel.close()
