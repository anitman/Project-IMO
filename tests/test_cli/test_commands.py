"""Tests for CLI commands."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from imo.cli.main import cli


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


class TestVersion:
    def test_version_flag(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output


class TestCategories:
    def test_categories_lists_all(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["categories"])
        assert result.exit_code == 0
        assert "llm" in result.output
        assert "vision_generation" in result.output
        assert "audio_speech_recognition" in result.output
        assert "video_generation" in result.output


class TestStats:
    def test_stats_shows_metrics(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["stats"])
        assert result.exit_code == 0
        assert "Total Projects" in result.output
        assert "Active Nodes" in result.output


class TestNodeStatus:
    def test_node_status(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["node", "status"])
        assert result.exit_code == 0
        assert "IMO Node Status" in result.output
        assert "Version" in result.output


class TestToolkit:
    def test_toolkit_list(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["toolkit", "list"])
        assert result.exit_code == 0
        assert "HF Trainer" in result.output
        assert "Unsloth" in result.output
        assert "Diffusers" in result.output

    def test_toolkit_info_hf_trainer(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["toolkit", "info", "hf_trainer"])
        assert result.exit_code == 0
        assert "HF Trainer" in result.output
        assert "transformers" in result.output
        assert "Modes" in result.output

    def test_toolkit_info_unknown(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["toolkit", "info", "nonexistent"])
        assert result.exit_code == 1

    def test_toolkit_install_known(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["toolkit", "install", "unsloth"])
        assert result.exit_code == 0
        assert "Unsloth" in result.output

    def test_toolkit_install_unknown(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["toolkit", "install", "nonexistent"])
        assert result.exit_code == 1
        assert "Unknown toolkit" in result.output


class TestProjectCreate:
    def test_create_with_all_args(self, runner: CliRunner, tmp_path: Path) -> None:
        project_dir = str(tmp_path / "test-project")
        result = runner.invoke(cli, [
            "project", "create",
            "--title", "Test LLM",
            "--category", "llm",
            "--mode", "from_scratch",
            "--max-steps", "5000",
            "--project-dir", project_dir,
        ])
        assert result.exit_code == 0
        assert "Test LLM" in result.output
        assert "scaffolded" in result.output
        assert Path(project_dir, "imo.toml").exists()
        assert Path(project_dir, "configs").is_dir()
        assert Path(project_dir, "scripts").is_dir()

    def test_create_with_toolkit(self, runner: CliRunner, tmp_path: Path) -> None:
        project_dir = str(tmp_path / "diffusion-project")
        result = runner.invoke(cli, [
            "project", "create",
            "--title", "My Diffusion",
            "--category", "vision_generation",
            "--mode", "lora",
            "--toolkit", "diffusers",
            "--project-dir", project_dir,
        ])
        assert result.exit_code == 0
        assert "diffusers" in result.output
        assert Path(project_dir, "imo.toml").exists()

    def test_create_with_base_model(self, runner: CliRunner, tmp_path: Path) -> None:
        project_dir = str(tmp_path / "finetune-project")
        result = runner.invoke(cli, [
            "project", "create",
            "--title", "Fine Tune Test",
            "--category", "llm_chat",
            "--mode", "lora",
            "--base-model", "meta-llama/Llama-3-8B",
            "--project-dir", project_dir,
        ])
        assert result.exit_code == 0
        assert "meta-llama/Llama-3-8B" in result.output

    def test_create_scaffolds_imo_toml(self, runner: CliRunner, tmp_path: Path) -> None:
        project_dir = str(tmp_path / "toml-project")
        runner.invoke(cli, [
            "project", "create",
            "--title", "TOML Test",
            "--category", "llm",
            "--mode", "from_scratch",
            "--project-dir", project_dir,
        ])
        toml_path = Path(project_dir, "imo.toml")
        assert toml_path.exists()
        content = toml_path.read_text()
        assert 'title = "TOML Test"' in content
        assert 'category = "llm"' in content
        assert 'training_mode = "from_scratch"' in content
        assert 'toolkit = "hf_trainer"' in content


class TestProjectInfo:
    def test_project_info(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["project", "info", "abc123"])
        assert result.exit_code == 0
        assert "abc123" in result.output


class TestProjectJoin:
    def test_join_compute(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["project", "join", "abc123", "--compute"])
        assert result.exit_code == 0
        assert "READY" in result.output


class TestTrainStart:
    def test_train_start(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, [
            "train", "start", "abc123",
            "--batch-size", "16",
            "--lr", "0.001",
        ])
        assert "abc123" in result.output
        assert "16" in result.output
        assert "Distributed Training" in result.output

    def test_train_start_pipeline_mode(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, [
            "train", "start", "abc123",
            "--parallelism", "pipeline_parallel",
        ])
        assert "pipeline" in result.output.lower()
