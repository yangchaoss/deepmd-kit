#!/usr/bin/env python3
"""Small real-device forward/backward dispatch probe for the candidate op."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def max_abs(a, b) -> float:
    import torch

    return float((a - b).abs().max().detach().cpu())


def load_adapter_install():
    path = Path(__file__).resolve().parents[1] / "runner" / "runtime_adapter.py"
    spec = importlib.util.spec_from_file_location("dpa4c_nano_runtime_adapter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load tracked adapter: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.install


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--shared-object", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, object] = {
        "schema_version": "dpa4c-contest-runtime.dispatch-probe.v1",
        "status": "FAIL",
        "shared_object": str(a.shared_object.resolve()),
        "shared_object_sha256": sha256(a.shared_object),
        "negative_control": {},
    }
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/PPU device is unavailable")
        install = load_adapter_install()
        adapter_info = install(a.shared_object)
        device = torch.device("cuda:0")
        dtype = torch.float32
        g_e = torch.tensor(
            [[0.5, -1.0, 2.0], [1.5, 0.25, -0.5], [-0.75, 2.0, 0.125], [0.25, -0.5, 1.25]],
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        edge_vec = torch.tensor(
            [[1.0, 0.0, 0.5], [-0.25, 2.0, 0.0], [0.0, 0.5, -1.0], [2.0, -1.0, 0.25]],
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        edge_index = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 2]], device=device, dtype=torch.int64)
        edge_mask = torch.tensor([True, True, False, True], device=device, dtype=torch.bool)
        edge_frame = torch.tensor([0, 0, 1, 1], device=device, dtype=torch.int64)
        node_capacity = 3
        frame_count = 2
        force, atom_virial, virial = torch.ops.dpa4c_contest.edge_force_virial(
            g_e, edge_vec, edge_index, edge_mask, edge_frame, node_capacity, frame_count
        )

        # Reference forward path uses the same public tensor operations as the
        # official Python scatter, but remains independent of the candidate op.
        ref_force = torch.zeros((node_capacity, 3), device=device, dtype=dtype)
        ref_force = ref_force.index_add(0, edge_index[1], g_e * edge_mask[:, None].to(dtype))
        ref_force = ref_force.index_add(0, edge_index[0], -g_e * edge_mask[:, None].to(dtype))
        w_edge = -(g_e[:, :, None] * edge_vec[:, None, :]) * edge_mask[:, None, None].to(dtype)
        ref_atom = torch.zeros((node_capacity, 3, 3), device=device, dtype=dtype).index_add(
            0, edge_index[0], w_edge
        )
        ref_virial = torch.zeros((frame_count, 3, 3), device=device, dtype=dtype).index_add(
            0, edge_frame, w_edge
        )
        torch.cuda.synchronize()
        forward_diff = {
            "force_max_abs": max_abs(force, ref_force),
            "atom_virial_max_abs": max_abs(atom_virial, ref_atom),
            "virial_max_abs": max_abs(virial, ref_virial),
        }

        weights_force = torch.tensor(
            [[1.0, -0.5, 0.25], [0.5, 0.75, -1.0], [-0.25, 0.125, 0.5]], device=device, dtype=dtype
        )
        weights_atom = torch.arange(node_capacity * 9, device=device, dtype=dtype).reshape(node_capacity, 3, 3) / 7.0
        weights_virial = torch.arange(frame_count * 9, device=device, dtype=dtype).reshape(frame_count, 3, 3) / 11.0
        candidate_loss = (force * weights_force).sum() + (atom_virial * weights_atom).sum() + (virial * weights_virial).sum()
        candidate_grad_g, candidate_grad_vec = torch.autograd.grad(candidate_loss, (g_e, edge_vec), retain_graph=True)

        ref_loss = (ref_force * weights_force).sum() + (ref_atom * weights_atom).sum() + (ref_virial * weights_virial).sum()
        ref_grad_g, ref_grad_vec = torch.autograd.grad(ref_loss, (g_e, edge_vec), retain_graph=True)
        torch.cuda.synchronize()
        backward_diff = {
            "g_e_max_abs": max_abs(candidate_grad_g, ref_grad_g),
            "edge_vec_max_abs": max_abs(candidate_grad_vec, ref_grad_vec),
        }

        # A deliberately perturbed input is a negative control for the harness:
        # the observed output must change; this is not an anti-cheat claim.
        perturbed = g_e.detach().clone()
        perturbed[0, 0] += 1.0e-2
        perturbed_out = torch.ops.dpa4c_contest.edge_force_virial(
            perturbed, edge_vec.detach(), edge_index, edge_mask, edge_frame, node_capacity, frame_count
        )[0]
        negative_delta = max_abs(perturbed_out, force.detach())
        result["adapter"] = adapter_info
        result["device"] = {
            "name": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda_version": torch.version.cuda,
        }
        result["dispatch"] = {
            "operator": "dpa4c_contest::edge_force_virial",
            "device_dispatch_observed": True,
            "forward": forward_diff,
            "backward": backward_diff,
        }
        result["negative_control"] = {
            "input": "g_e[0,0] += 1e-2",
            "output_force_max_delta": negative_delta,
            "detected": negative_delta > 0.0,
        }
        result["stream_contract"] = "DEFAULT_STREAM_ONLY; producer/consumer stream test not implemented"
        result["kernel_trace"] = {
            "status": "NOT_AVAILABLE",
            "gap": "This probe demonstrates live PPU dispatch but has no profiler/kernel-trace artifact.",
        }
        result["status"] = "PASS" if (
            max(forward_diff.values()) <= 1.0e-6
            and max(backward_diff.values()) <= 1.0e-6
            and bool(result["negative_control"]["detected"])
        ) else "FAIL"
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
    a.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
