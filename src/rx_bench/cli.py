"""``rxbench`` — the rx-bench command line.

A thin wrapper that (a) imports :mod:`rx_bench`, which registers the
``medical_reception`` domain with tau2 and applies the env-var config patch,
then (b) dispatches:

* harness subcommands are handled here:

    rxbench merge       merge data/v1/cases/*.json into tasks.json (+ splits)
    rxbench vacuity     reject vacuous / unsatisfiable cases
    rxbench score       results.json -> scorecard.json + readable table
    rxbench mutate      mutation-test the suite against weakened policies
    rxbench diversity   report distinct behaviours vs copy-paste clusters
    rxbench repair      re-run simulations lost to infrastructure errors

* anything else (``run``, ``view``, ``domain``, ...) is passed through to the
  stock ``tau2`` CLI, so ``rxbench run --domain medical_reception ...`` is
  exactly ``tau2 run ...`` with the domain guaranteed to be registered.
"""

from __future__ import annotations

import sys
from typing import Optional

import rx_bench  # noqa: F401  (import side effect: registration + config patch)

_HARNESS_COMMANDS = {
    "merge": ("rx_bench.harness.merge_cases", "main"),
    "vacuity": ("rx_bench.harness.vacuity", "main"),
    "score": ("rx_bench.harness.scorecard", "main"),
    "mutate": ("rx_bench.harness.mutate", "main"),
    "diversity": ("rx_bench.harness.diversity", "main"),
    "repair": ("rx_bench.harness.repair", "main"),
}


def main(argv: Optional[list[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    if args and args[0] in _HARNESS_COMMANDS:
        module_name, func_name = _HARNESS_COMMANDS[args[0]]
        import importlib

        module = importlib.import_module(module_name)
        func = getattr(module, func_name)
        rest = args[1:]
        import inspect

        if inspect.signature(func).parameters:
            return int(func(rest) or 0)
        # A few harness mains take no argv and read sys.argv themselves.
        sys.argv = [f"rxbench-{args[0]}", *rest]
        return int(func() or 0)

    # Pass everything else through to the tau2 CLI (domain already registered).
    from tau2.cli import main as tau2_main

    sys.argv = ["tau2", *args]
    return tau2_main() or 0


if __name__ == "__main__":
    raise SystemExit(main())
