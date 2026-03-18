"""Command-line interface for IMO."""

from __future__ import annotations

import json
import uuid

import click

from imo import __version__
from imo.data import CodeSecurityScanner, DatasetLinter, ModelCategory, get_model_category_description


@click.group()
@click.version_option(version=__version__)
def cli() -> None:
    """IMO — Initial Model Offering Platform.

    Decentralized AI training: propose models, contribute datasets,
    donate compute, earn rewards.
    """


# ── Data Commands ──────────────────────────────────────────────


@cli.group()
def data() -> None:
    """Dataset operations: lint, scan, inspect."""


@data.command()
@click.argument("dataset_path", type=click.Path(exists=True))
@click.option("--min-score", default=0.7, help="Minimum quality score")
@click.option("--security/--no-security", default=True, help="Run security scan")
def lint(dataset_path: str, min_score: float, security: bool) -> None:
    """Lint a dataset for quality and security."""
    linter = DatasetLinter(min_quality_score=min_score)
    result = linter.lint_file(dataset_path)

    click.echo(f"  Quality Score : {result.quality_score:.2f}")
    click.echo(f"  Quality Level : {result.quality_level.value}")
    click.echo(f"  Samples       : {result.stats.get('num_samples', 'N/A')}")

    if result.issues:
        click.echo("\n  Issues:")
        for issue in result.issues:
            click.echo(f"    ✗ {issue}")

    if result.warnings:
        click.echo("\n  Warnings:")
        for warning in result.warnings:
            click.echo(f"    ⚠ {warning}")

    if security:
        scanner = CodeSecurityScanner()
        sec_result = scanner.scan_file(dataset_path)
        if not sec_result.is_safe:
            click.echo("\n  Security Issues:")
            for issue in sec_result.issues:
                click.echo(f"    [{issue.threat_level.value}] {issue.message}")
        else:
            click.echo("\n  Security: PASS")

    if result.is_acceptable(min_score):
        click.echo(f"\n  Result: PASS (>= {min_score})")
    else:
        click.echo(f"\n  Result: FAIL (below {min_score})")
        raise SystemExit(1)


# ── Model Category Commands ───────────────────────────────────


@cli.command("categories")
def list_categories() -> None:
    """List supported model categories and their dataset requirements."""
    for category in ModelCategory:
        desc = get_model_category_description(category)
        click.echo(f"  {category.value:<30s} {desc}")


# ── Node Commands ─────────────────────────────────────────────


@cli.group()
def node() -> None:
    """Node operations: connect, status, peers."""


@node.command()
@click.argument("peer_id")
def connect(peer_id: str) -> None:
    """Connect to a peer in the network."""
    click.echo(f"Connecting to peer: {peer_id}...")
    click.echo("Use `imo node status` to check connection state.")


@node.command()
def status() -> None:
    """Show node status."""
    import torch

    click.echo("IMO Node Status")
    click.echo("=" * 40)
    click.echo(f"  Version : {__version__}")
    click.echo(f"  CUDA    : {'available' if torch.cuda.is_available() else 'not available'}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_mem / (1024**3)
            click.echo(f"  GPU {i}   : {name} ({mem:.1f} GB)")


# ── Project Commands (future: backed by DHT / chain) ─────────


@cli.group()
def project() -> None:
    """Project operations: create, list, contribute."""


@project.command("create")
@click.option("--title", required=True, help="Project title")
@click.option("--arch", required=True, help="Model architecture (e.g. llama, dit, unet)")
@click.option("--category", required=True, help="Model category (e.g. llm, vision_generation)")
@click.option("--mode", default="from_scratch", help="Training mode: from_scratch|full_fine_tune|lora")
@click.option("--base-model", default=None, help="Base model ID for fine-tuning")
@click.option("--max-steps", default=100000, help="Maximum training steps")
def create_project(
    title: str,
    arch: str,
    category: str,
    mode: str,
    base_model: str | None,
    max_steps: int,
) -> None:
    """Create a new training project."""
    project_id = str(uuid.uuid4())[:8]
    click.echo(f"  Project ID    : {project_id}")
    click.echo(f"  Title         : {title}")
    click.echo(f"  Architecture  : {arch}")
    click.echo(f"  Category      : {category}")
    click.echo(f"  Training Mode : {mode}")
    if base_model:
        click.echo(f"  Base Model    : {base_model}")
    click.echo(f"  Max Steps     : {max_steps}")
    click.echo(f"\n  Status: DRAFT — run `imo project open {project_id}` to accept datasets.")


@project.command("contribute")
@click.argument("project_id")
@click.argument("dataset_path", type=click.Path(exists=True))
@click.option("--license", "data_license", default="apache-2.0", help="Dataset license")
def contribute_dataset(project_id: str, dataset_path: str, data_license: str) -> None:
    """Contribute a dataset to a project."""
    linter = DatasetLinter()
    result = linter.lint_file(dataset_path)

    click.echo(f"  Dataset Quality : {result.quality_score:.2f} ({result.quality_level.value})")

    if not result.is_acceptable():
        click.echo("  Contribution REJECTED: dataset below quality threshold.")
        raise SystemExit(1)

    click.echo(f"  Contributing to project {project_id}...")
    click.echo(f"  License: {data_license}")
    click.echo("  Status: SUBMITTED (pending verification)")


# ── Training Commands ─────────────────────────────────────────


@cli.group()
def train() -> None:
    """Training operations: start, join, monitor."""


@train.command("start")
@click.argument("project_id")
@click.option("--peers", default="", help="Comma-separated bootstrap peer addresses")
@click.option("--batch-size", default=32, help="Local batch size")
@click.option("--lr", default=1e-4, help="Learning rate")
def start_training(project_id: str, peers: str, batch_size: int, lr: float) -> None:
    """Start or join distributed training for a project."""
    peer_list = [p.strip() for p in peers.split(",") if p.strip()]
    click.echo(f"  Project       : {project_id}")
    click.echo(f"  Batch Size    : {batch_size}")
    click.echo(f"  Learning Rate : {lr}")
    click.echo(f"  Peers         : {len(peer_list)} bootstrap peers")
    click.echo("\n  Initializing hivemind DHT and joining training swarm...")


if __name__ == "__main__":
    cli()
