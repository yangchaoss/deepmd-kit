"""Candidate session using the public ASE O(N) graph builder."""
from __future__ import annotations

import os

os.environ.setdefault("DP_CUDA_INFER", "2")
os.environ.setdefault("DP_PPU_FUSED_FITTING", "enabled")

from ase.calculators.calculator import all_changes


class DeepMDCandidateSession:
    def __init__(self, model: str) -> None:
        from deepmd.infer.deep_pot import DeepPot

        self._dp = DeepPot(model, neighbor_graph_method="ase")
        self._type_dict = {
            name: index for index, name in enumerate(self._dp.get_type_map())
        }

    def evaluate(self, atoms) -> dict[str, object]:
        import numpy as np

        coords = atoms.get_positions().reshape([1, -1])
        cells = atoms.get_cell().reshape([1, -1]) if bool(np.any(atoms.pbc)) else None
        atom_types = [self._type_dict[symbol] for symbol in atoms.get_chemical_symbols()]
        energy, forces, virial = self._dp.eval(
            coords=coords,
            cells=cells,
            atom_types=atom_types,
        )[:3]
        energy_value = np.asarray(energy)[0]
        forces_value = np.asarray(forces)[0]
        virial_value = np.asarray(virial)[0].reshape(3, 3)
        result: dict[str, object] = {
            "energy": energy_value,
            "forces": forces_value,
            "virial": virial_value,
        }
        if cells is not None:
            stress = -0.5 * (virial_value + virial_value.T) / atoms.get_volume()
            result["stress"] = stress.flat[[0, 4, 8, 5, 2, 1]]
        return result


def create_session(*, model: str) -> DeepMDCandidateSession:
    return DeepMDCandidateSession(model)
