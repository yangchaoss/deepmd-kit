"""Minimal candidate session contract.

The organizer owns inputs and the evaluation loop.  Candidate code only creates
the implementation object used for one evaluation.
"""
from __future__ import annotations

import os
os.environ.setdefault("DP_CUDA_INFER", "2")


from ase.calculators.calculator import all_changes


class DeepMDCandidateSession:
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
