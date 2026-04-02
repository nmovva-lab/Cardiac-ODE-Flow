#!/usr/bin/env python3
"""
Cardio-ODE-Flow — main entry point

Usage:
    # Train from scratch
    python main.py train

    # Resume from a checkpoint
    python main.py train --resume checkpoints/epoch_0050_auroc_0.8900.pt

    # Evaluate a checkpoint on the test set
    python main.py eval --checkpoint checkpoints/epoch_0050_auroc_0.8900.pt

    # Quick sanity check (runs one forward pass with synthetic data, no real dataset needed)
    python main.py sanity
"""

import argparse
import sys
import torch

from configs.config import get_config


def cmd_train(args):
    from training.trainer import train
    cfg = get_config()
    # Override data paths from CLI if provided
    if args.ptbxl:
        cfg.data.ptbxl_path = args.ptbxl
    if args.georgia:
        cfg.data.georgia_path = args.georgia
    if args.chapman:
        cfg.data.chapman_path = args.chapman
    train(cfg, resume_from=args.resume)


def cmd_eval(args):
    from training.evaluate import full_evaluation
    full_evaluation(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
    )


def cmd_sanity(args):
    """
    Runs a single forward + backward pass with random data.
    Confirms the full pipeline is wired correctly before training.
    Does NOT require any real ECG data.
    """
    print("\n── Cardio-ODE-Flow sanity check ──────────────────────────────")
    cfg = get_config()

    # Reduce sizes for fast check
    cfg.data.sequence_length = 1000
    cfg.model.graph.gcn_hidden_dim = 32
    cfg.model.graph.gcn_output_dim = 64
    cfg.model.graph.gcn_layers = 2
    cfg.model.gru.hidden_dim = 64
    cfg.model.ode.latent_dim = 64
    cfg.model.ode.ode_hidden_dim = 128
    cfg.model.flow.latent_dim = 64
    cfg.model.flow.num_coupling_layers = 2
    cfg.model.flow.hidden_dim = 64

    # Sync dims
    cfg.model.gru.input_dim = cfg.model.graph.gcn_output_dim

    from training.trainer import get_device
    from models.model import CardioODEFlow
    from training.loss import HybridLoss

    device = get_device(cfg)
    print(f"Device: {device}")

    B = 2  # tiny batch
    model = CardioODEFlow(cfg).to(device)
    params = model.count_parameters()
    print(f"Parameters: {params['total']:,}")

    ecg = torch.randn(B, 12, cfg.data.sequence_length, device=device)
    age = torch.rand(B, device=device)
    sex = torch.randint(0, 2, (B,), device=device).float()
    labels = torch.randint(0, 2, (B, cfg.model.num_classes), device=device).float()

    print("Running forward pass...")
    out = model(ecg, age, sex, return_traj=True)

    print(f"  logits:     {out['logits'].shape}")
    print(f"  probs:      {out['probs'].shape}")
    print(f"  confidence: {out['confidence'].shape}  values: {out['confidence'].tolist()}")
    print(f"  h_final:    {out['h_final'].shape}")
    print(f"  h_traj:     {out['h_traj'].shape}  (T, B, latent_dim)")
    print(f"  z:          {out['z'].shape}")
    print(f"  log_prob:   {out['log_prob'].tolist()}")

    loss_fn = HybridLoss()
    loss_fn.set_epoch(10)
    loss, metrics = loss_fn(out["logits"], labels, out["log_prob"])
    print(f"\nLoss breakdown:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\nRunning backward pass...")
    loss.backward()
    print("  ✓ Backward pass succeeded")

    # Check all parameters have gradients
    no_grad = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    if no_grad:
        print(f"  ⚠ Parameters without gradient: {no_grad}")
    else:
        print("  ✓ All parameters received gradients")

    # Test predict (no_grad mode)
    preds = model.predict(ecg, age, sex)
    print(f"\nPredict output:")
    print(f"  probs:       {preds['probs'].shape}")
    print(f"  predictions: {preds['predictions'].shape}")
    print(f"  confidence:  {preds['confidence'].tolist()}")

    print("\n✓ Sanity check passed!\n")


def main():
    parser = argparse.ArgumentParser(description="Cardio-ODE-Flow")
    sub = parser.add_subparsers(dest="cmd")

    # train
    p_train = sub.add_parser("train")
    p_train.add_argument("--resume", type=str, default=None)
    p_train.add_argument("--ptbxl", type=str, default=None)
    p_train.add_argument("--georgia", type=str, default=None)
    p_train.add_argument("--chapman", type=str, default=None)

    # eval
    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--checkpoint", type=str, required=True)
    p_eval.add_argument("--output_dir", type=str, default="eval_results")

    # sanity
    sub.add_parser("sanity")

    args = parser.parse_args()

    if args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "eval":
        cmd_eval(args)
    elif args.cmd == "sanity":
        cmd_sanity(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
