import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

# Ensure repo root is importable when script is launched outside project root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def build_args(obs=11, preds=120):
    # Mirrors train.py defaults used by TrajAirNet.
    return SimpleNamespace(
        obs=obs,
        preds=preds,
        input_channels=3,
        tcn_channel_size=256,
        tcn_layers=2,
        tcn_kernels=4,
        num_context_input_c=2,
        num_context_output_c=7,
        cnn_kernels=2,
        gat_heads=16,
        graph_hidden=256,
        dropout=0.05,
        alpha=0.2,
        cvae_hidden=128,
        cvae_channel_size=128,
        cvae_layers=2,
        mlp_layer=32,
        k=4,
        num_samples=15,
        traj_dim=3,
        agent_num=3,
        k_retrieve=20,
        n_clusters=3,
    )


def main():
    parser = argparse.ArgumentParser(description="Export TrajAirNet to ONNX")
    parser.add_argument("--ckpt", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--out", default="trajairnet.onnx", help="Output ONNX path")
    parser.add_argument("--obs", type=int, default=11)
    parser.add_argument("--preds", type=int, default=120)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--agents", type=int, default=7)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    import torch
    from model.trajairnet import TrajAirNet

    class OnnxInferenceWrapper(torch.nn.Module):
        """Wraps TrajAirNet.inference into a 2-input ONNX-exportable interface."""

        def __init__(self, model: TrajAirNet):
            super().__init__()
            self.model = model

        def forward(self, obs_traj, route_priors):
            # Build placeholder args to satisfy the original inference signature.
            bsz, agents, _, _ = obs_traj.shape
            pred_placeholder = torch.zeros(
                (bsz, agents, 3, 12), dtype=obs_traj.dtype, device=obs_traj.device
            )
            adj_placeholder = torch.ones((agents, agents), dtype=obs_traj.dtype, device=obs_traj.device)
            context_placeholder = torch.zeros(
                (bsz, agents, 2, 11), dtype=obs_traj.dtype, device=obs_traj.device
            )
            return self.model.inference(
                obs_traj,
                pred_placeholder,
                adj_placeholder,
                context_placeholder,
                route_priors=route_priors,
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_args = build_args(obs=args.obs, preds=args.preds)

    model = TrajAirNet(model_args).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    wrapper = OnnxInferenceWrapper(model).to(device).eval()

    obs_traj = torch.randn(args.batch, args.agents, 3, args.obs, device=device)
    route_priors = torch.randn(args.batch, args.agents, 3, 12, 3, device=device)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.onnx.export(
        wrapper,
        (obs_traj, route_priors),
        args.out,
        opset_version=args.opset,
        input_names=["obs_traj", "route_priors"],
        output_names=["generated_y"],
        dynamic_axes={
            "obs_traj": {0: "batch", 1: "agents"},
            "route_priors": {0: "batch", 1: "agents"},
            "generated_y": {0: "batch_agents"},
        },
    )
    print(f"Exported ONNX to: {args.out}")


if __name__ == "__main__":
    main()
