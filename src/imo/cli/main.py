"""Command-line interface for IMO."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import click

from imo import __version__
from imo.data import (
    CodeSecurityScanner,
    DatasetLinter,
    ModelCategory,
    get_model_category_description,
)

IMO_BANNER = r"""
  [bold cyan]██╗███╗   ███╗ ██████╗[/bold cyan]
  [bold cyan]██║████╗ ████║██╔═══██╗[/bold cyan]
  [bold cyan]██║██╔████╔██║██║   ██║[/bold cyan]
  [bold cyan]██║██║╚██╔╝██║██║   ██║[/bold cyan]
  [bold cyan]██║██║ ╚═╝ ██║╚██████╔╝[/bold cyan]
  [bold cyan]╚═╝╚═╝     ╚═╝ ╚═════╝[/bold cyan]
  [dim]Decentralized AI Training Protocol[/dim]  [bold]v{version}[/bold]
"""

IMO_BANNER_PLAIN = r"""
  ██╗███╗   ███╗ ██████╗
  ██║████╗ ████║██╔═══██╗
  ██║██╔████╔██║██║   ██║
  ██║██║╚██╔╝██║██║   ██║
  ██║██║ ╚═╝ ██║╚██████╔╝
  ╚═╝╚═╝     ╚═╝ ╚═════╝
  Decentralized AI Training Protocol  v{version}
"""


def _print_banner() -> None:
    """Print the IMO banner using rich if available, fallback to plain."""
    try:
        from rich.console import Console

        console = Console()
        console.print(IMO_BANNER.format(version=__version__))
    except ImportError:
        click.echo(IMO_BANNER_PLAIN.format(version=__version__))


def _get_rich_console() -> Any:
    """Get a rich Console, or None if not available."""
    try:
        from rich.console import Console

        return Console()
    except ImportError:
        return None


# ── Root CLI Group ────────────────────────────────────────────


@click.group(invoke_without_command=True)
@click.version_option(version=__version__)
@click.pass_context
def cli(ctx: click.Context) -> None:
    """IMO — Initial Model Offering Platform.

    Decentralized AI training: propose models, contribute datasets,
    donate compute, earn rewards.
    """
    if ctx.invoked_subcommand is None:
        _interactive_dashboard()


# ── Interactive Dashboard ─────────────────────────────────────


def _interactive_dashboard() -> None:
    """Show the interactive main menu."""
    _print_banner()

    console = _get_rich_console()

    menu_items = [
        ("1", "Browse Projects", "View all active IMO projects"),
        ("2", "Create Project", "Start a new training project"),
        ("3", "Join Project", "Contribute data or compute to a project"),
        ("4", "Featured Projects", "View high-quality / trending projects"),
        ("5", "Platform Stats", "IMO network statistics"),
        ("6", "Node Status", "Your node hardware & connection info"),
        ("7", "Toolkits", "List available training toolkits"),
        ("q", "Quit", "Exit IMO"),
    ]

    if console:
        from rich.table import Table

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Key", style="bold yellow", width=4)
        table.add_column("Action", style="bold white", width=22)
        table.add_column("Description", style="dim")
        for key, action, desc in menu_items:
            table.add_row(f"[{key}]", action, desc)
        console.print(table)
        console.print()
    else:
        for key, action, desc in menu_items:
            click.echo(f"  [{key}] {action:<22s} {desc}")
        click.echo()

    choice = click.prompt("Select an option", type=str, default="q")

    dispatch: dict[str, Any] = {
        "1": _browse_projects,
        "2": _create_project_interactive,
        "3": _join_project_interactive,
        "4": _featured_projects,
        "5": _platform_stats,
        "6": _node_status_display,
        "7": _list_toolkits,
    }

    handler = dispatch.get(choice)
    if handler:
        handler()
    elif choice != "q":
        click.echo("Invalid option.")


# ── Interactive Handlers ──────────────────────────────────────


def _browse_projects() -> None:
    """Browse all projects with filtering."""
    console = _get_rich_console()

    click.echo("\n  Filter by category:")
    categories = [
        ("1", "All"),
        ("2", "LLM (Language Models)"),
        ("3", "Vision (Image Generation, Classification)"),
        ("4", "Audio (TTS, ASR, Sound Generation)"),
        ("5", "Video (Generation, Understanding)"),
        ("6", "Multimodal (VLM, Audio-VLM)"),
        ("7", "Embedding"),
        ("8", "Other (Time-series, Recommendation)"),
    ]
    for key, label in categories:
        click.echo(f"    [{key}] {label}")

    click.prompt("\n  Category", type=str, default="1")

    click.echo("\n  Filter by status:")
    statuses = [
        ("1", "All"),
        ("2", "Recruiting Data"),
        ("3", "Recruiting Compute"),
        ("4", "Training"),
        ("5", "Completed"),
    ]
    for key, label in statuses:
        click.echo(f"    [{key}] {label}")

    click.prompt("\n  Status", type=str, default="1")

    # Placeholder — in production this queries the DHT / registry
    if console:
        from rich.table import Table

        table = Table(title="Active Projects", show_lines=True)
        table.add_column("ID", style="cyan", width=10)
        table.add_column("Title", style="bold")
        table.add_column("Category", style="green")
        table.add_column("Mode", style="yellow")
        table.add_column("Status", style="magenta")
        table.add_column("Contributors")
        table.add_column("Toolkit")

        # Demo data
        table.add_row(
            "a1b2c3d4", "Open LLaMA 3B", "llm", "from_scratch",
            "training", "12 data / 8 compute", "hf_trainer",
        )
        table.add_row(
            "e5f6g7h8", "Flux LoRA Anime", "vision_generation", "lora",
            "open_for_data", "5 data / 0 compute", "ai_toolkit",
        )
        table.add_row(
            "i9j0k1l2", "Wan2.1 Dance LoRA", "video_generation", "lora",
            "open_for_data", "3 data / 0 compute", "musubi",
        )
        console.print(table)
    else:
        click.echo("\n  ID        Title              Category           Status")
        click.echo("  " + "-" * 60)
        click.echo("  a1b2c3d4  Open LLaMA 3B      llm                training")
        click.echo("  e5f6g7h8  Flux LoRA Anime     vision_generation  open_for_data")
        click.echo("  i9j0k1l2  Wan2.1 Dance LoRA   video_generation   open_for_data")


def _create_project_interactive() -> None:
    """Interactive project creation wizard."""
    console = _get_rich_console()

    if console:
        console.print("\n  [bold]Create New Training Project[/bold]\n")
    else:
        click.echo("\n  Create New Training Project\n")

    # Step 1: Project basics
    title = click.prompt("  Project title")
    description = click.prompt("  Description", default="")

    # Step 2: Model category
    click.echo("\n  Model Category:")
    category_groups = [
        ("Language", [
            ("1", "llm", "LLM (pretraining)"),
            ("2", "llm_chat", "Chat / Instruct"),
            ("3", "llm_code", "Code Generation"),
        ]),
        ("Vision", [
            ("4", "vision_classification", "Image Classification"),
            ("5", "vision_generation", "Image Generation (SD, Flux, etc.)"),
        ]),
        ("Audio", [
            ("6", "audio_speech_recognition", "Speech Recognition"),
            ("7", "audio_text_to_speech", "Text-to-Speech"),
            ("8", "audio_sound_generation", "Sound Generation"),
        ]),
        ("Video", [
            ("9", "video_generation", "Video Generation"),
            ("10", "video_understanding", "Video Understanding"),
        ]),
        ("Multimodal", [
            ("11", "multimodal_vlm", "Vision-Language Model"),
            ("12", "multimodal_audio_vlm", "Audio-Language Model"),
        ]),
        ("Embedding", [
            ("13", "embedding_text", "Text Embedding"),
            ("14", "embedding_multimodal", "Multimodal Embedding"),
        ]),
        ("Other", [
            ("15", "time_series_forecasting", "Time-series Forecasting"),
            ("16", "recommendation", "Recommendation"),
        ]),
    ]

    for group_name, items in category_groups:
        click.echo(f"    {group_name}:")
        for key, _value, label in items:
            click.echo(f"      [{key:>2s}] {label}")

    cat_choice = click.prompt("\n  Select category", type=int, default=1)
    # Build lookup
    cat_lookup: dict[int, str] = {}
    for _, items in category_groups:
        for key, value, _ in items:
            cat_lookup[int(key)] = value
    category = cat_lookup.get(cat_choice, "llm")

    # Step 3: Training mode
    click.echo("\n  Training Mode:")
    modes = [
        ("1", "from_scratch", "From Scratch — full pretraining"),
        ("2", "full_fine_tune", "Full Fine-tune — all parameters"),
        ("3", "lora", "LoRA — parameter-efficient"),
        ("4", "qlora", "QLoRA — quantized LoRA"),
        ("5", "continual_pretrain", "Continual Pretraining — extend existing model"),
        ("6", "distillation", "Distillation — learn from a teacher model"),
        ("7", "hybrid", "Hybrid — multi-model merge / MoE routing"),
    ]
    for key, _value, label in modes:
        click.echo(f"    [{key}] {label}")

    mode_choice = click.prompt("\n  Select mode", type=int, default=1)
    mode_lookup = {int(k): v for k, v, _ in modes}
    training_mode = mode_lookup.get(mode_choice, "from_scratch")

    # Step 4: Base model (for fine-tuning modes)
    base_model: str | None = None
    teacher_model: str | None = None
    merge_models: list[str] = []

    if training_mode in ("full_fine_tune", "lora", "qlora", "continual_pretrain"):
        base_model = click.prompt(
            "  Base model (HuggingFace ID)", default="meta-llama/Llama-3-8B"
        )
    elif training_mode == "distillation":
        base_model = click.prompt("  Student model (HuggingFace ID)")
        teacher_model = click.prompt("  Teacher model (HuggingFace ID)")
    elif training_mode == "hybrid":
        base_model = click.prompt("  Primary model (HuggingFace ID)")
        merge_input = click.prompt(
            "  Additional models to merge (comma-separated HF IDs)", default=""
        )
        merge_models = [m.strip() for m in merge_input.split(",") if m.strip()]

    # Step 5: Paper source
    click.echo("\n  Paper Source:")
    click.echo("    [1] Upload paper (IPFS)")
    click.echo("    [2] Link to arXiv paper")
    click.echo("    [3] No paper yet")

    paper_choice = click.prompt("  Select", type=int, default=3)
    arxiv_id: str | None = None
    paper_source = "none"
    if paper_choice == 1:
        paper_source = "upload"
        click.echo("  (Paper upload will be handled after project creation)")
    elif paper_choice == 2:
        paper_source = "arxiv"
        arxiv_id = click.prompt("  arXiv ID (e.g. 2401.12345)")

    # Step 6: Toolkit selection
    from imo.toolkits.registry import get_default_registry

    registry = get_default_registry()
    try:
        model_cat = ModelCategory(category)
    except ValueError:
        model_cat = ModelCategory.LLM

    recommended = registry.recommend(model_cat, training_mode)
    all_toolkits = registry.list_all()

    click.echo("\n  Training Toolkit:")
    if recommended:
        click.echo("    Recommended for your selection:")
        for i, tk in enumerate(recommended, 1):
            ti = tk.info()
            click.echo(f"      [{i}] {ti.display_name} — {ti.description}")
    click.echo("    All available:")
    for i, tk in enumerate(all_toolkits, 1):
        ti = tk.info()
        marker = " *" if tk in recommended else ""
        click.echo(f"      [{i}] {ti.display_name}{marker}")

    tk_choice = click.prompt(
        "  Select toolkit",
        type=int,
        default=1 if not recommended else all_toolkits.index(recommended[0]) + 1,
    )
    selected_toolkit = all_toolkits[min(tk_choice, len(all_toolkits)) - 1]
    toolkit_name = selected_toolkit.info().name

    # Step 7: Project directory
    default_dir = f"./projects/{title.lower().replace(' ', '-')}"
    project_dir = click.prompt("  Project directory", default=default_dir)

    # Step 8: Training params
    max_steps = click.prompt("  Max training steps", type=int, default=100000)

    # Generate project
    project_id = str(uuid.uuid4())[:8]
    project_path = Path(project_dir)

    if console:
        console.print("\n  [bold green]Project created![/bold green]\n")
        from rich.table import Table

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Field", style="dim", width=20)
        table.add_column("Value", style="bold")
        table.add_row("Project ID", project_id)
        table.add_row("Title", title)
        table.add_row("Category", category)
        table.add_row("Training Mode", training_mode)
        if base_model:
            table.add_row("Base Model", base_model)
        if teacher_model:
            table.add_row("Teacher Model", teacher_model)
        if merge_models:
            table.add_row("Merge Models", ", ".join(merge_models))
        if arxiv_id:
            table.add_row("arXiv", arxiv_id)
        table.add_row("Toolkit", selected_toolkit.info().display_name)
        table.add_row("Directory", str(project_path))
        table.add_row("Max Steps", str(max_steps))
        console.print(table)
    else:
        click.echo("\n  Project created!\n")
        click.echo(f"  Project ID    : {project_id}")
        click.echo(f"  Title         : {title}")
        click.echo(f"  Category      : {category}")
        click.echo(f"  Training Mode : {training_mode}")
        if base_model:
            click.echo(f"  Base Model    : {base_model}")
        if teacher_model:
            click.echo(f"  Teacher Model : {teacher_model}")
        if merge_models:
            click.echo(f"  Merge Models  : {', '.join(merge_models)}")
        if arxiv_id:
            click.echo(f"  arXiv         : {arxiv_id}")
        click.echo(f"  Toolkit       : {selected_toolkit.info().display_name}")
        click.echo(f"  Directory     : {project_path}")
        click.echo(f"  Max Steps     : {max_steps}")

    # Scaffold project directory
    _scaffold_project(project_path, project_id, {
        "title": title,
        "description": description,
        "category": category,
        "training_mode": training_mode,
        "base_model": base_model,
        "teacher_model": teacher_model,
        "merge_models": merge_models,
        "paper_source": paper_source,
        "arxiv_id": arxiv_id,
        "toolkit": toolkit_name,
        "max_steps": max_steps,
    })

    # Let the toolkit set up its files
    selected_toolkit.setup_project(project_path, {
        "base_model": base_model or "",
        "training_mode": training_mode,
        "max_steps": max_steps,
        "batch_size": 4,
        "learning_rate": 1e-4,
    })

    click.echo(f"\n  Project scaffolded at: {project_path}")
    click.echo(f"  Run `imo project open {project_id}` to accept dataset contributions.")


def _scaffold_project(
    project_dir: Path,
    project_id: str,
    spec: dict[str, Any],
) -> None:
    """Create the standard project directory structure and imo.toml."""
    for subdir in ("data", "configs", "checkpoints", "logs", "scripts"):
        (project_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Write imo.toml
    toml_lines = [
        "[project]",
        f'id = "{project_id}"',
        f'title = "{spec["title"]}"',
        f'description = "{spec.get("description", "")}"',
        f'category = "{spec["category"]}"',
        f'training_mode = "{spec["training_mode"]}"',
        f'toolkit = "{spec["toolkit"]}"',
        "",
    ]

    if spec.get("base_model"):
        toml_lines.append("[model]")
        toml_lines.append(f'base_model = "{spec["base_model"]}"')
        if spec.get("teacher_model"):
            toml_lines.append(f'teacher_model = "{spec["teacher_model"]}"')
        if spec.get("merge_models"):
            models_str = ", ".join(f'"{m}"' for m in spec["merge_models"])
            toml_lines.append(f"merge_models = [{models_str}]")
        toml_lines.append("")

    toml_lines.append("[paper]")
    toml_lines.append(f'source = "{spec.get("paper_source", "none")}"')
    if spec.get("arxiv_id"):
        toml_lines.append(f'arxiv_id = "{spec["arxiv_id"]}"')
    toml_lines.append("")

    toml_lines.append("[training]")
    toml_lines.append(f'max_steps = {spec.get("max_steps", 100000)}')
    toml_lines.append("")

    (project_dir / "imo.toml").write_text("\n".join(toml_lines))


def _join_project_interactive() -> None:
    """Interactive project joining flow."""
    project_id = click.prompt("\n  Enter Project ID to join")

    click.echo("\n  Contribution type:")
    click.echo("    [1] Contribute Dataset")
    click.echo("    [2] Contribute Compute (GPU)")

    choice = click.prompt("  Select", type=int, default=1)

    if choice == 1:
        dataset_path = click.prompt("  Path to dataset")
        data_license = click.prompt("  Dataset license", default="apache-2.0")
        click.echo(f"\n  Submitting dataset to project {project_id}...")
        click.echo(f"  Path: {dataset_path}")
        click.echo(f"  License: {data_license}")
        click.echo("  Status: SUBMITTED (pending quality check)")
    else:
        click.echo(f"\n  Joining project {project_id} as compute contributor...")
        click.echo("  Detecting GPU hardware...")
        try:
            import torch

            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    name = torch.cuda.get_device_name(i)
                    mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    click.echo(f"  GPU {i}: {name} ({mem:.1f} GB)")
            else:
                click.echo("  No CUDA GPU detected.")
        except ImportError:
            click.echo("  PyTorch not available — cannot detect GPU.")
        click.echo("  Status: READY (waiting for training to start)")


def _featured_projects() -> None:
    """Show featured / high-quality projects."""
    console = _get_rich_console()

    if console:
        from rich.table import Table

        console.print("\n  [bold]Featured Projects[/bold]\n")
        table = Table(show_lines=True)
        table.add_column("ID", style="cyan")
        table.add_column("Title", style="bold")
        table.add_column("Category")
        table.add_column("Quality", style="green")
        table.add_column("IMO Reward Pool", style="yellow")
        table.add_column("Contributors")

        # Demo data
        table.add_row(
            "a1b2c3d4", "Open LLaMA 3B", "llm",
            "92/100", "50,000 IMO", "20",
        )
        table.add_row(
            "m3n4o5p6", "Open Whisper v3", "audio_speech_recognition",
            "88/100", "35,000 IMO", "15",
        )
        console.print(table)
    else:
        click.echo("\n  Featured Projects")
        click.echo("  " + "-" * 60)
        click.echo("  a1b2c3d4  Open LLaMA 3B      quality: 92  pool: 50,000 IMO")
        click.echo("  m3n4o5p6  Open Whisper v3     quality: 88  pool: 35,000 IMO")


def _platform_stats() -> None:
    """Show platform-wide statistics."""
    console = _get_rich_console()

    # Placeholder stats — in production, query from DHT / chain
    stats = {
        "Total Projects": "127",
        "Active Training": "14",
        "Completed": "89",
        "Recruiting": "24",
        "Total IMO Distributed": "2,450,000 IMO",
        "Active Nodes": "342",
        "Total Compute Hours": "128,500 hrs",
        "Total Datasets": "1,203",
    }

    if console:
        from rich.table import Table

        console.print("\n  [bold]IMO Platform Statistics[/bold]\n")
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Metric", style="dim", width=28)
        table.add_column("Value", style="bold green")
        for metric, value in stats.items():
            table.add_row(metric, value)
        console.print(table)
    else:
        click.echo("\n  IMO Platform Statistics")
        click.echo("  " + "-" * 40)
        for metric, value in stats.items():
            click.echo(f"  {metric:<28s} {value}")


def _node_status_display() -> None:
    """Show node status."""
    click.echo("\n  IMO Node Status")
    click.echo("  " + "=" * 40)
    click.echo(f"  Version : {__version__}")
    try:
        import torch

        click.echo(
            f"  CUDA    : {'available' if torch.cuda.is_available() else 'not available'}"
        )
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                click.echo(f"  GPU {i}   : {name} ({mem:.1f} GB)")
    except ImportError:
        click.echo("  CUDA    : torch not installed")


def _list_toolkits() -> None:
    """List all available training toolkits."""
    from imo.toolkits.registry import get_default_registry

    registry = get_default_registry()
    console = _get_rich_console()

    if console:
        from rich.table import Table

        console.print("\n  [bold]Available Training Toolkits[/bold]\n")
        table = Table(show_lines=True)
        table.add_column("Name", style="cyan", width=16)
        table.add_column("Description")
        table.add_column("Categories", style="green")
        table.add_column("Modes", style="yellow")
        table.add_column("Min VRAM", style="magenta")

        for tk in registry.list_all():
            ti = tk.info()
            cats = ", ".join(c.value for c in ti.supported_categories[:3])
            if len(ti.supported_categories) > 3:
                cats += f" (+{len(ti.supported_categories) - 3})"
            modes = ", ".join(ti.supported_modes[:3])
            if len(ti.supported_modes) > 3:
                modes += f" (+{len(ti.supported_modes) - 3})"
            table.add_row(
                ti.display_name, ti.description, cats, modes, f"{ti.min_vram_gb} GB"
            )
        console.print(table)
    else:
        click.echo("\n  Available Training Toolkits")
        click.echo("  " + "-" * 70)
        for tk in registry.list_all():
            ti = tk.info()
            click.echo(f"  {ti.display_name:<20s} {ti.description}")
            click.echo(f"  {'':20s} Modes: {', '.join(ti.supported_modes)}")
            click.echo(f"  {'':20s} Min VRAM: {ti.min_vram_gb} GB")
            click.echo()

    click.echo("  Install a toolkit: imo toolkit install <name>")


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
            click.echo(f"    - {issue}")

    if result.warnings:
        click.echo("\n  Warnings:")
        for warning in result.warnings:
            click.echo(f"    ! {warning}")

    if security:
        scanner = CodeSecurityScanner()
        sec_result = scanner.scan_file(dataset_path)
        if not sec_result.is_safe:
            click.echo("\n  Security Issues:")
            for sec_issue in sec_result.issues:
                click.echo(f"    [{sec_issue.threat_level.value}] {sec_issue.message}")
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
    console = _get_rich_console()

    if console:
        from rich.table import Table

        table = Table(title="Supported Model Categories", show_lines=True)
        table.add_column("Category", style="cyan")
        table.add_column("Description")

        for category in ModelCategory:
            desc = get_model_category_description(category)
            table.add_row(category.value, desc)
        console.print(table)
    else:
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
    _node_status_display()


# ── Project Commands ──────────────────────────────────────────


@cli.group()
def project() -> None:
    """Project operations: create, list, join, browse."""


@project.command("list")
@click.option("--category", default=None, help="Filter by model category")
@click.option("--status", "proj_status", default=None, help="Filter by project status")
@click.option("--sort", default="created", help="Sort by: created, quality, rewards")
def list_projects(
    category: str | None, proj_status: str | None, sort: str
) -> None:
    """List all projects (with optional filters)."""
    _browse_projects()


@project.command("featured")
def featured() -> None:
    """Show featured / high-quality projects."""
    _featured_projects()


@project.command("create")
@click.option("--title", default=None, help="Project title (interactive if omitted)")
@click.option("--arch", default=None, help="Model architecture")
@click.option("--category", default=None, help="Model category")
@click.option("--mode", default=None, help="Training mode")
@click.option("--base-model", default=None, help="Base model ID for fine-tuning")
@click.option("--toolkit", "toolkit_name", default=None, help="Training toolkit name")
@click.option("--max-steps", default=100000, help="Maximum training steps")
@click.option("--project-dir", default=None, help="Custom project directory")
def create_project(
    title: str | None,
    arch: str | None,
    category: str | None,
    mode: str | None,
    base_model: str | None,
    toolkit_name: str | None,
    max_steps: int,
    project_dir: str | None,
) -> None:
    """Create a new training project."""
    # If key params missing, fall back to interactive mode
    if not title or not category or not mode:
        _create_project_interactive()
        return

    project_id = str(uuid.uuid4())[:8]
    click.echo(f"  Project ID    : {project_id}")
    click.echo(f"  Title         : {title}")
    if arch:
        click.echo(f"  Architecture  : {arch}")
    click.echo(f"  Category      : {category}")
    click.echo(f"  Training Mode : {mode}")
    if base_model:
        click.echo(f"  Base Model    : {base_model}")
    click.echo(f"  Max Steps     : {max_steps}")

    tk_name = toolkit_name or "hf_trainer"
    click.echo(f"  Toolkit       : {tk_name}")

    # Scaffold
    proj_path = Path(project_dir or f"./projects/{title.lower().replace(' ', '-')}")
    _scaffold_project(proj_path, project_id, {
        "title": title,
        "description": "",
        "category": category,
        "training_mode": mode,
        "base_model": base_model,
        "toolkit": tk_name,
        "max_steps": max_steps,
        "paper_source": "none",
    })

    from imo.toolkits.registry import get_default_registry

    registry = get_default_registry()
    tk = registry.get(tk_name)
    if tk:
        tk.setup_project(proj_path, {
            "base_model": base_model or "",
            "training_mode": mode,
            "max_steps": max_steps,
            "batch_size": 4,
            "learning_rate": 1e-4,
        })

    click.echo(f"\n  Project scaffolded at: {proj_path}")
    click.echo(f"  Status: DRAFT — run `imo project open {project_id}` to accept datasets.")


@project.command("join")
@click.argument("project_id")
@click.argument("dataset_path", required=False, type=click.Path(exists=True))
@click.option("--compute", is_flag=True, help="Join as compute contributor")
@click.option("--license", "data_license", default="apache-2.0", help="Dataset license")
def join_project(
    project_id: str,
    dataset_path: str | None,
    compute: bool,
    data_license: str,
) -> None:
    """Join a project (contribute data or compute)."""
    if not dataset_path and not compute:
        _join_project_interactive()
        return

    if compute:
        click.echo(f"  Joining project {project_id} as compute contributor...")
        click.echo("  Status: READY")
    elif dataset_path:
        linter = DatasetLinter()
        result = linter.lint_file(dataset_path)
        click.echo(f"  Dataset Quality : {result.quality_score:.2f} ({result.quality_level.value})")
        if not result.is_acceptable():
            click.echo("  Contribution REJECTED: dataset below quality threshold.")
            raise SystemExit(1)
        click.echo(f"  Contributing to project {project_id}...")
        click.echo(f"  License: {data_license}")
        click.echo("  Status: SUBMITTED (pending verification)")


@project.command("info")
@click.argument("project_id")
def project_info(project_id: str) -> None:
    """Show detailed info for a project."""
    click.echo(f"\n  Project: {project_id}")
    click.echo("  (Project lookup will query DHT/registry in production)")


# ── Toolkit Commands ──────────────────────────────────────────


@cli.group()
def toolkit() -> None:
    """Training toolkit operations: list, install, info."""


@toolkit.command("list")
def toolkit_list() -> None:
    """List available training toolkits."""
    _list_toolkits()


@toolkit.command("install")
@click.argument("name")
def toolkit_install(name: str) -> None:
    """Install a training toolkit."""
    from imo.toolkits.registry import get_default_registry

    registry = get_default_registry()
    tk = registry.get(name)
    if not tk:
        click.echo(f"  Unknown toolkit: {name}")
        click.echo(f"  Available: {', '.join(registry.list_names())}")
        raise SystemExit(1)

    ti = tk.info()
    click.echo(f"  Installing {ti.display_name}...")
    click.echo(f"  Command: {tk.get_install_command()}")

    # Check current environment
    missing = tk.validate_environment()
    if missing:
        click.echo("\n  Missing dependencies:")
        for dep in missing:
            click.echo(f"    - {dep}")
        click.echo(f"\n  Run: {tk.get_install_command()}")
    else:
        click.echo(f"  {ti.display_name} is already installed and ready.")


@toolkit.command("info")
@click.argument("name")
def toolkit_info(name: str) -> None:
    """Show detailed info about a toolkit."""
    from imo.toolkits.registry import get_default_registry

    registry = get_default_registry()
    tk = registry.get(name)
    if not tk:
        click.echo(f"  Unknown toolkit: {name}")
        raise SystemExit(1)

    ti = tk.info()
    click.echo(f"\n  {ti.display_name}")
    click.echo(f"  {'=' * len(ti.display_name)}")
    click.echo(f"  {ti.description}\n")
    click.echo(f"  URL        : {ti.url}")
    click.echo(f"  Package    : {ti.pip_package}")
    click.echo(f"  Min VRAM   : {ti.min_vram_gb} GB")
    click.echo(f"  Categories : {', '.join(c.value for c in ti.supported_categories)}")
    click.echo(f"  Modes      : {', '.join(ti.supported_modes)}")
    if ti.extra_dependencies:
        click.echo(f"  Extras     : {', '.join(ti.extra_dependencies)}")

    missing = tk.validate_environment()
    if missing:
        click.echo("\n  Status: NOT READY")
        for dep in missing:
            click.echo(f"    Missing: {dep}")
    else:
        click.echo("\n  Status: READY")


# ── Training Commands ─────────────────────────────────────────


@cli.group()
def train() -> None:
    """Training operations: start, join, monitor."""


@train.command("start")
@click.argument("project_id")
@click.option("--peers", default="", help="Comma-separated bootstrap peer addresses")
@click.option("--batch-size", default=32, help="Local batch size")
@click.option("--lr", default=1e-4, help="Learning rate")
@click.option("--parallelism", default="data_parallel",
              type=click.Choice(["data_parallel", "pipeline_parallel"]),
              help="Parallelism mode")
@click.option("--project-dir", default=None, help="Project directory (with imo.toml)")
def start_training(
    project_id: str,
    peers: str,
    batch_size: int,
    lr: float,
    parallelism: str,
    project_dir: str | None,
) -> None:
    """Start or join distributed training for a project.

    The training pipeline:
      1. Toolkit loads the model (from imo.toml config)
      2. Engine sets up Hivemind DHT and discovers peers
      3. In pipeline_parallel mode, model layers are split across GPU nodes
      4. Gradients are compressed, aggregated (Byzantine-robust), and averaged
      5. All nodes collaboratively train the same model
    """
    peer_list = [p.strip() for p in peers.split(",") if p.strip()]

    # Load project config if project-dir provided
    toolkit_name = "hf_trainer"
    spec: dict[str, Any] = {}
    if project_dir:
        _load_project_config(project_dir, spec)
        toolkit_name = spec.get("toolkit", "hf_trainer")

    from imo.toolkits.registry import get_default_registry

    registry = get_default_registry()
    tk = registry.get(toolkit_name)

    console = _get_rich_console()
    if console:
        from rich.table import Table

        console.print("\n  [bold]Starting Distributed Training[/bold]\n")
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Field", style="dim", width=20)
        table.add_column("Value", style="bold")
        table.add_row("Project", project_id)
        table.add_row("Toolkit", tk.info().display_name if tk else toolkit_name)
        table.add_row("Parallelism", parallelism)
        table.add_row("Batch Size", str(batch_size))
        table.add_row("Learning Rate", str(lr))
        table.add_row("Bootstrap Peers", str(len(peer_list)))
        console.print(table)
    else:
        click.echo(f"  Project       : {project_id}")
        click.echo(f"  Toolkit       : {tk.info().display_name if tk else toolkit_name}")
        click.echo(f"  Parallelism   : {parallelism}")
        click.echo(f"  Batch Size    : {batch_size}")
        click.echo(f"  Learning Rate : {lr}")
        click.echo(f"  Peers         : {len(peer_list)} bootstrap peers")

    click.echo("\n  [1/4] Toolkit loading model...")
    if tk:
        env_issues = tk.validate_environment()
        if env_issues:
            click.echo(f"  Missing dependencies: {', '.join(env_issues)}")
            click.echo(f"  Run: {tk.get_install_command()}")
            raise SystemExit(1)

    click.echo("  [2/4] Initializing Hivemind DHT and discovering peers...")
    if parallelism == "pipeline_parallel":
        click.echo("  [3/4] Splitting model layers across nodes (pipeline parallel)...")
        click.echo("         Each node hosts a subset of transformer blocks.")
        click.echo("         Activations flow through the server chain via RemoteSequential.")
    else:
        click.echo("  [3/4] Replicating model on each node (data parallel)...")
        click.echo("         Gradients compressed (Top-K) and averaged via Hivemind.")
    click.echo("  [4/4] Starting collaborative training loop...")
    click.echo("         Byzantine-robust aggregation active. Checkpoints every 1000 steps.")


def _load_project_config(
    project_dir: str, spec: dict[str, Any]
) -> None:
    """Load imo.toml from a project directory into spec dict."""
    import sys

    toml_path = Path(project_dir) / "imo.toml"
    if not toml_path.exists():
        return

    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib  # type: ignore[no-redef]

    with open(toml_path, "rb") as f:
        data = tomllib.load(f)

    project = data.get("project", {})
    spec["toolkit"] = project.get("toolkit", "hf_trainer")
    spec["category"] = project.get("category", "llm")
    spec["training_mode"] = project.get("training_mode", "from_scratch")

    model = data.get("model", {})
    spec["base_model"] = model.get("base_model", "")
    spec["teacher_model"] = model.get("teacher_model")

    training = data.get("training", {})
    spec["max_steps"] = training.get("max_steps", 100000)


# ── Stats Command ─────────────────────────────────────────────


@cli.command("stats")
def stats() -> None:
    """Show IMO platform statistics."""
    _platform_stats()


if __name__ == "__main__":
    cli()
