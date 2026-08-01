"""Live conversation with the medical_reception agent.

The suite grades the agent by having an LLM play the caller. This package puts
a *person* in that seat instead: the same agent, the same nineteen tools, the
same ``policy.md``, answering whatever you actually say.

Two channels, one conversation. :class:`TextChannel` reads stdin and prints;
:class:`VoiceChannel` records the microphone, transcribes it, and speaks the
reply. Everything above the channel is identical, which is the point --- the
voice demo is not a second implementation of the agent, it is the same run with
different I/O, so a transcript captured either way is scoreable by the same
scorers.
"""
