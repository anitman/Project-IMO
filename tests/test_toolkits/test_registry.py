"""Tests for the training toolkit registry and built-in adapters."""

from __future__ import annotations

from pathlib import Path

import pytest

from imo.data.dataset_spec import ModelCategory
from imo.toolkits.base import ToolkitCapability, ToolkitInfo
from imo.toolkits.registry import ToolkitRegistry, get_default_registry


class TestToolkitRegistry:
    """Test ToolkitRegistry operations."""

    def test_register_and_get(self) -> None:
        registry = ToolkitRegistry()
        default = get_default_registry()
        hf = default.get("hf_trainer")
        assert hf is not None
        registry.register(hf)
        assert registry.get("hf_trainer") is hf

    def test_get_unknown_returns_none(self) -> None:
        registry = ToolkitRegistry()
        assert registry.get("nonexistent") is None

    def test_list_all(self) -> None:
        registry = get_default_registry()
        all_toolkits = registry.list_all()
        assert len(all_toolkits) == 6

    def test_list_names(self) -> None:
        registry = get_default_registry()
        names = registry.list_names()
        assert "hf_trainer" in names
        assert "unsloth" in names
        assert "axolotl" in names
        assert "diffusers" in names
        assert "musubi" in names
        assert "ai_toolkit" in names

    def test_find_for_category_llm(self) -> None:
        registry = get_default_registry()
        llm_toolkits = registry.find_for_category(ModelCategory.LLM)
        names = [t.info().name for t in llm_toolkits]
        assert "hf_trainer" in names
        assert "unsloth" in names
        assert "axolotl" in names

    def test_find_for_category_vision_generation(self) -> None:
        registry = get_default_registry()
        vision_toolkits = registry.find_for_category(ModelCategory.VISION_GENERATION)
        names = [t.info().name for t in vision_toolkits]
        assert "diffusers" in names
        assert "ai_toolkit" in names

    def test_find_for_category_video_generation(self) -> None:
        registry = get_default_registry()
        video_toolkits = registry.find_for_category(ModelCategory.VIDEO_GENERATION)
        names = [t.info().name for t in video_toolkits]
        assert "diffusers" in names
        assert "musubi" in names

    def test_find_for_mode_lora(self) -> None:
        registry = get_default_registry()
        lora_toolkits = registry.find_for_mode(ToolkitCapability.LORA)
        names = [t.info().name for t in lora_toolkits]
        assert "hf_trainer" in names
        assert "unsloth" in names

    def test_recommend(self) -> None:
        registry = get_default_registry()
        recs = registry.recommend(ModelCategory.LLM, ToolkitCapability.LORA)
        names = [t.info().name for t in recs]
        assert "hf_trainer" in names
        assert "unsloth" in names

    def test_recommend_no_match(self) -> None:
        registry = get_default_registry()
        recs = registry.recommend(ModelCategory.RECOMMENDATION, ToolkitCapability.DISTILLATION)
        assert len(recs) == 0 or all(
            t.supports_category(ModelCategory.RECOMMENDATION)
            and t.supports_mode(ToolkitCapability.DISTILLATION)
            for t in recs
        )


class TestBuiltinAdapters:
    """Test each built-in adapter's info, config, and project setup."""

    @pytest.fixture()
    def registry(self) -> ToolkitRegistry:
        return get_default_registry()

    @pytest.mark.parametrize(
        "name",
        ["hf_trainer", "unsloth", "axolotl", "diffusers", "musubi", "ai_toolkit"],
    )
    def test_info_fields(self, registry: ToolkitRegistry, name: str) -> None:
        tk = registry.get(name)
        assert tk is not None
        ti = tk.info()
        assert isinstance(ti, ToolkitInfo)
        assert ti.name == name
        assert len(ti.display_name) > 0
        assert len(ti.description) > 0
        assert len(ti.url) > 0
        assert len(ti.pip_package) > 0
        assert len(ti.supported_categories) > 0
        assert len(ti.supported_modes) > 0
        assert ti.min_vram_gb >= 1

    @pytest.mark.parametrize(
        "name",
        ["hf_trainer", "unsloth", "axolotl", "diffusers", "musubi", "ai_toolkit"],
    )
    def test_validate_environment_returns_list(
        self, registry: ToolkitRegistry, name: str
    ) -> None:
        tk = registry.get(name)
        assert tk is not None
        result = tk.validate_environment()
        assert isinstance(result, list)

    @pytest.mark.parametrize(
        "name",
        ["hf_trainer", "unsloth", "axolotl", "diffusers", "musubi", "ai_toolkit"],
    )
    def test_prepare_config(self, registry: ToolkitRegistry, name: str) -> None:
        tk = registry.get(name)
        assert tk is not None
        config = tk.prepare_config({
            "base_model": "test-model",
            "training_mode": "lora",
            "max_steps": 1000,
            "batch_size": 2,
            "learning_rate": 1e-4,
        })
        assert isinstance(config, dict)
        assert "output_dir" in config

    @pytest.mark.parametrize(
        "name",
        ["hf_trainer", "unsloth", "axolotl", "diffusers", "musubi", "ai_toolkit"],
    )
    def test_setup_project(
        self, registry: ToolkitRegistry, name: str, tmp_path: Path
    ) -> None:
        tk = registry.get(name)
        assert tk is not None
        project_dir = tmp_path / f"test-{name}"
        tk.setup_project(project_dir, {
            "base_model": "test-model",
            "training_mode": "lora",
            "max_steps": 1000,
            "batch_size": 2,
            "learning_rate": 1e-4,
        })
        assert (project_dir / "configs").is_dir()
        assert (project_dir / "scripts").is_dir()
        assert (project_dir / "scripts" / "train.sh").exists()

    @pytest.mark.parametrize(
        "name",
        ["hf_trainer", "unsloth", "axolotl", "diffusers", "musubi", "ai_toolkit"],
    )
    def test_get_train_command(
        self, registry: ToolkitRegistry, name: str, tmp_path: Path
    ) -> None:
        tk = registry.get(name)
        assert tk is not None
        cmd = tk.get_train_command(tmp_path)
        assert isinstance(cmd, list)
        assert len(cmd) >= 2
        assert cmd[0] == "bash"

    @pytest.mark.parametrize(
        "name",
        ["hf_trainer", "unsloth", "axolotl", "diffusers", "musubi", "ai_toolkit"],
    )
    def test_get_install_command(self, registry: ToolkitRegistry, name: str) -> None:
        tk = registry.get(name)
        assert tk is not None
        cmd = tk.get_install_command()
        assert cmd.startswith("pip install")

    def test_hf_trainer_lora_config(self, registry: ToolkitRegistry) -> None:
        tk = registry.get("hf_trainer")
        assert tk is not None
        config = tk.prepare_config({"training_mode": "lora", "lora_r": 32})
        assert config["use_peft"] is True
        assert config["lora_r"] == 32

    def test_hf_trainer_qlora_config(self, registry: ToolkitRegistry) -> None:
        tk = registry.get("hf_trainer")
        assert tk is not None
        config = tk.prepare_config({"training_mode": "qlora"})
        assert config["use_peft"] is True
        assert config["load_in_4bit"] is True

    def test_hf_trainer_distillation_config(self, registry: ToolkitRegistry) -> None:
        tk = registry.get("hf_trainer")
        assert tk is not None
        config = tk.prepare_config({
            "training_mode": "distillation",
            "teacher_model": "big-model",
        })
        assert config["teacher_model"] == "big-model"

    def test_unsloth_qlora_config(self, registry: ToolkitRegistry) -> None:
        tk = registry.get("unsloth")
        assert tk is not None
        config = tk.prepare_config({"training_mode": "qlora"})
        assert config["load_in_4bit"] is True

    def test_diffusers_lora_config(self, registry: ToolkitRegistry) -> None:
        tk = registry.get("diffusers")
        assert tk is not None
        config = tk.prepare_config({"training_mode": "lora", "lora_r": 8})
        assert config["use_lora"] is True
        assert config["lora_r"] == 8

    def test_axolotl_lora_config(self, registry: ToolkitRegistry) -> None:
        tk = registry.get("axolotl")
        assert tk is not None
        config = tk.prepare_config({"training_mode": "lora", "lora_r": 64})
        assert config["adapter"] == "lora"
        assert config["lora_r"] == 64

    def test_axolotl_qlora_config(self, registry: ToolkitRegistry) -> None:
        tk = registry.get("axolotl")
        assert tk is not None
        config = tk.prepare_config({"training_mode": "qlora"})
        assert config["adapter"] == "qlora"

    def test_supports_category(self, registry: ToolkitRegistry) -> None:
        hf = registry.get("hf_trainer")
        assert hf is not None
        assert hf.supports_category(ModelCategory.LLM)
        assert not hf.supports_category(ModelCategory.VIDEO_GENERATION)

    def test_supports_mode(self, registry: ToolkitRegistry) -> None:
        unsloth = registry.get("unsloth")
        assert unsloth is not None
        assert unsloth.supports_mode(ToolkitCapability.LORA)
        assert not unsloth.supports_mode(ToolkitCapability.DISTILLATION)
