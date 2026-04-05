"""
Compare ecs-ml vs PyTorch MNIST MLP training.
Runs both, collects JSON metrics, prints side-by-side comparison.
"""

import json
import subprocess
import sys
import os

import typer

app = typer.Typer()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
VENV_PYTHON = os.path.join(SCRIPT_DIR, ".venv", "bin", "python3")


def run_rust():
    print("=" * 60)
    print("Running ecs-ml (Rust)...")
    print("=" * 60)
    result = subprocess.run(
        ["cargo", "run", "--release", "--bin", "mnist_mlp", "--", "--json"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    sys.stderr.write(result.stderr)
    if result.returncode != 0:
        print(f"Rust build/run failed:\n{result.stderr}", file=sys.stderr)
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"Failed to parse Rust JSON output:\n{result.stdout}", file=sys.stderr)
        return None


def run_pytorch():
    print("=" * 60)
    print("Running PyTorch...")
    print("=" * 60)
    python = VENV_PYTHON if os.path.exists(VENV_PYTHON) else sys.executable
    result = subprocess.run(
        [python, os.path.join(SCRIPT_DIR, "mnist_mlp.py")],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    sys.stderr.write(result.stderr)
    if result.returncode != 0:
        print(f"PyTorch failed:\n{result.stderr}", file=sys.stderr)
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"Failed to parse PyTorch JSON output:\n{result.stdout}", file=sys.stderr)
        return None


def print_comparison(rust, pytorch):
    print()
    print("=" * 70)
    print("  COMPARISON: ecs-ml (Rust/Bevy ECS) vs PyTorch")
    print("=" * 70)
    print()

    print(f"{'Metric':<25} {'ecs-ml':>15} {'PyTorch':>15} {'Ratio':>10}")
    print("-" * 70)

    print(f"{'Architecture':<25} {'MLP 784-256-128-10':>15} {'MLP 784-256-128-10':>15}")
    print(f"{'Parameters':<25} {rust['total_params']:>15,} {pytorch['total_params']:>15,}")
    print()

    rust_total = rust["total_time_ms"]
    pt_total = pytorch["total_time_ms"]
    ratio = rust_total / pt_total if pt_total > 0 else float("inf")
    print(f"{'Total time (ms)':<25} {rust_total:>15,.0f} {pt_total:>15,.0f} {ratio:>9.1f}x")

    rust_epochs = rust["epochs"]
    pt_epochs = pytorch["epochs"]
    rust_avg_epoch = sum(e["time_ms"] for e in rust_epochs) / len(rust_epochs)
    pt_avg_epoch = sum(e["time_ms"] for e in pt_epochs) / len(pt_epochs)
    ratio = rust_avg_epoch / pt_avg_epoch if pt_avg_epoch > 0 else float("inf")
    print(f"{'Avg epoch time (ms)':<25} {rust_avg_epoch:>15,.0f} {pt_avg_epoch:>15,.0f} {ratio:>9.1f}x")

    rust_mem = rust.get("peak_memory_mb", 0)
    pt_mem = pytorch.get("peak_memory_mb", 0)
    if rust_mem > 0 and pt_mem > 0:
        ratio = rust_mem / pt_mem if pt_mem > 0 else float("inf")
        print(f"{'Peak RSS (MB)':<25} {rust_mem:>15.1f} {pt_mem:>15.1f} {ratio:>9.1f}x")
    print()

    print(f"{'Epoch':<7} {'--- ecs-ml ---':>20} {'--- PyTorch ---':>22}")
    print(f"{'':7} {'Loss':>7} {'Train%':>8} {'Test%':>7} {'Loss':>8} {'Train%':>8} {'Test%':>7}")
    print("-" * 70)

    for i in range(min(len(rust_epochs), len(pt_epochs))):
        r = rust_epochs[i]
        p = pt_epochs[i]
        print(
            f"  {i:<5} {r['loss']:>7.4f} {r['train_acc']:>7.1f}% {r['test_acc']:>6.1f}%"
            f" {p['loss']:>8.4f} {p['train_acc']:>7.1f}% {p['test_acc']:>6.1f}%"
        )

    print()
    r_final = rust_epochs[-1]
    p_final = pt_epochs[-1]
    diff = r_final["test_acc"] - p_final["test_acc"]
    print(f"{'Final test accuracy':<25} {r_final['test_acc']:>14.1f}% {p_final['test_acc']:>14.1f}%  (diff: {diff:+.1f}%)")
    print()

    output = {"rust": rust, "pytorch": pytorch}
    out_path = os.path.join(PROJECT_ROOT, "python", "comparison_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Full results saved to: {out_path}")


@app.command()
def main(
    skip_rust: bool = typer.Option(False, help="Skip Rust run, use cached results"),
    skip_pytorch: bool = typer.Option(False, help="Skip PyTorch run, use cached results"),
):
    """Run ecs-ml and PyTorch MNIST MLP training, compare results side-by-side."""
    cached_path = os.path.join(PROJECT_ROOT, "python", "comparison_results.json")

    rust_results = None
    pytorch_results = None

    if not skip_rust:
        rust_results = run_rust()
    elif os.path.exists(cached_path):
        with open(cached_path) as f:
            cached = json.load(f)
            rust_results = cached.get("rust")

    if not skip_pytorch:
        pytorch_results = run_pytorch()
    elif os.path.exists(cached_path):
        with open(cached_path) as f:
            cached = json.load(f)
            pytorch_results = cached.get("pytorch")

    if rust_results and pytorch_results:
        print_comparison(rust_results, pytorch_results)
    else:
        if rust_results:
            print("\necs-ml results:")
            print(json.dumps(rust_results, indent=2))
        if pytorch_results:
            print("\nPyTorch results:")
            print(json.dumps(pytorch_results, indent=2))
        if not rust_results and not pytorch_results:
            print("No results to compare.", file=sys.stderr)
            raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
