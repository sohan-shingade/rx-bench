# Third-party notices

## τ²-bench (tau2-bench)

This benchmark is built on top of **τ²-bench** by Sierra Research
(<https://github.com/sierra-research/tau2-bench>), used as a dependency pinned
at commit `363133a`. τ²-bench is licensed under the MIT License:

> MIT License
>
> Copyright (c) 2024 Sierra
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in
> all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
> FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
> IN THE SOFTWARE.

If you use this benchmark, please also cite the τ-bench line of work (see the
Citation section of the README):

- Yao et al., *τ-bench: A Benchmark for Tool-Agent-User Interaction in
  Real-World Domains* (2024), arXiv:2406.12045
- Barres et al., *τ²-Bench: Evaluating Conversational Agents in a Dual-Control
  Environment* (2025), arXiv:2506.07982
- Ray et al., *τ-Voice: Benchmarking Full-Duplex Voice Agents on Real-World
  Domains* (2026), arXiv:2603.13686

## FDA look-alike/sound-alike (Tall Man) list

The look-alike/sound-alike (LASA) drug confusion pairs used by the S1 suite
are informed by the U.S. Food and Drug Administration's list of established
drug names recommended to use Tall Man (mixed-case) lettering, and related
ISMP publications. As a work of the U.S. federal government, the FDA list is
in the public domain.

## ElevenLabs (planned audio corpus)

A future Tier-A audio corpus (released separately as a release asset, not in
this repository) may include speech synthesized with the ElevenLabs API.
Synthesized audio will be labeled as synthetic and distributed subject to the
ElevenLabs terms of service in force at generation time. The current shipped
corpus manifest (`src/rx_bench/audio/corpus/manifest.json`) describes renders
generated locally with macOS `say`.
