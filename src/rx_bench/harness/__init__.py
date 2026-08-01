"""Scoring, suite-maintenance, and meta-evaluation tools for the benchmark.

Everything here is importable and unit-testable offline (no API keys): the
scorers read a completed tau2 ``Results`` file, the suite tools read the case
files under ``data/<version>/``. Entry points are exposed via the ``rxbench``
CLI (merge, vacuity, score, mutate, diversity, repair).
"""
