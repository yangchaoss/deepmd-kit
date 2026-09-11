"""Official adapter that routes the live pt-expt graph EFS scatter through the candidate op."""
from __future__ import annotations

from pathlib import Path

import torch


_INSTALLED = False
_ORIGINAL = None


def _fake(g_e, edge_vec, edge_index, edge_mask, edge_frame, node_capacity: int, frame_count: int):
    return (
        torch.empty((node_capacity, 3), device="meta", dtype=g_e.dtype),
        torch.empty((node_capacity, 3, 3), device="meta", dtype=g_e.dtype),
        torch.empty((frame_count, 3, 3), device="meta", dtype=g_e.dtype),
    )


def _setup_context(ctx, inputs, output) -> None:
    g_e, edge_vec, edge_index, edge_mask, edge_frame, node_capacity, frame_count = inputs
    ctx.save_for_backward(g_e, edge_vec, edge_index, edge_mask, edge_frame)
    ctx.node_capacity = int(node_capacity)
    ctx.frame_count = int(frame_count)


def _backward(ctx, grad_force, grad_atom_virial, grad_virial):
    g_e, edge_vec, edge_index, edge_mask, edge_frame = ctx.saved_tensors
    edge_count = edge_index.shape[1]
    src = edge_index[0]
    dst = edge_index[1]
    frame = edge_frame
    if grad_force is None:
        grad_force = g_e.new_zeros((ctx.node_capacity, 3))
    grad_g = grad_force.index_select(0, dst) - grad_force.index_select(0, src)
    if grad_atom_virial is not None:
        atom = grad_atom_virial.index_select(0, src)
    else:
        atom = torch.zeros((edge_count, 3, 3), device=g_e.device, dtype=g_e.dtype)
    if grad_virial is not None:
        vir = grad_virial.index_select(0, frame)
    else:
        vir = torch.zeros_like(atom)
    total = atom + vir
    grad_g = grad_g - (total * edge_vec[:, None, :]).sum(dim=2)
    grad_edge_vec = -(g_e[:, :, None] * total).sum(dim=1)
    mask = edge_mask.to(g_e.dtype).reshape(-1, 1)
    return grad_g * mask, grad_edge_vec * mask, None, None, None, None, None


def install(shared_object: str | Path) -> dict[str, object]:
    """Load the candidate .so and replace only the live scatter seam."""
    global _INSTALLED, _ORIGINAL
    if _INSTALLED:
        return {"installed": True, "reused": True}
    torch.ops.load_library(str(Path(shared_object).resolve()))
    torch.library.register_fake("dpa4c_contest::edge_force_virial")(_fake)
    torch.library.register_autograd(
        "dpa4c_contest::edge_force_virial",
        _backward,
        setup_context=_setup_context,
    )
    import deepmd.pt_expt.model.edge_transform_output as edge_transform_output
    from deepmd.dpmodel.utils.neighbor_graph import frame_id_from_n_node
    from deepmd.pt.utils import env

    _ORIGINAL = edge_transform_output.edge_energy_deriv

    def contest_edge_energy_deriv(
        energy,
        edge_vec,
        edge_index,
        edge_mask,
        n_node,
        destination_order=None,
        destination_row_ptr=None,
        source_order=None,
        source_row_ptr=None,
        node_capacity=None,
        *,
        do_atomic_virial=False,
        create_graph=False,
        force_precision=None,
    ):
        # Training / double-backward is deliberately outside this prototype.
        if create_graph:
            return _ORIGINAL(
                energy, edge_vec, edge_index, edge_mask, n_node,
                destination_order, destination_row_ptr, source_order, source_row_ptr,
                node_capacity, do_atomic_virial=do_atomic_virial,
                create_graph=create_graph, force_precision=force_precision,
            )
        (g_e,) = torch.autograd.grad(
            energy.sum() if energy.dim() else energy,
            edge_vec,
            create_graph=False,
            retain_graph=True,
        )
        if force_precision is not None and g_e.dtype != force_precision:
            g_e = g_e.to(force_precision)
            edge_vec = edge_vec.to(force_precision)
        n_out = node_capacity if node_capacity is not None else int(n_node.sum())
        frame_id = frame_id_from_n_node(n_node, n_total=n_out)
        src_or_dst = edge_index[1] % n_out if n_out else edge_index[1]
        edge_frame = frame_id.index_select(0, src_or_dst)
        force, atom_virial, virial = torch.ops.dpa4c_contest.edge_force_virial(
            g_e.contiguous(), edge_vec.contiguous(), edge_index.contiguous(),
            edge_mask.contiguous(), edge_frame.contiguous(), int(n_out),
            int(n_node.shape[0]),
        )
        return force, (atom_virial if do_atomic_virial else None), virial

    edge_transform_output.edge_energy_deriv = contest_edge_energy_deriv
    _INSTALLED = True
    return {
        "installed": True,
        "reused": False,
        "seam": "deepmd.pt_expt.model.edge_transform_output.edge_energy_deriv",
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "dispatch": "dpa4c_contest::edge_force_virial",
        "training_create_graph": "falls back to official reference; prototype inference only",
    }
