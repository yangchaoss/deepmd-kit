"""Minimal candidate session contract.

The organizer owns inputs and the evaluation loop.  Candidate code only creates
the implementation object used for one evaluation.
"""
from __future__ import annotations

from ase.calculators.calculator import all_changes


SIMULATION_IMPLEMENTATION_ID = "contestant-simulation-20260917"


class DeepMDCandidateSession:
    implementation_id = SIMULATION_IMPLEMENTATION_ID

    def __init__(self, model: str) -> None:
        from deepmd.calculator import DP

        self._calculator = DP(model=model, nlist_backend="auto")

    def evaluate(self, atoms) -> dict[str, object]:
        self._calculator.calculate(
            atoms=atoms,
            properties=["energy", "forces", "virial", "stress"],
            system_changes=all_changes,
        )
        return dict(self._calculator.results)


def create_session(*, model: str) -> DeepMDCandidateSession:
    return DeepMDCandidateSession(model)
