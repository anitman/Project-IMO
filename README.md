# IMO — Initial Model Offering

```
  ██╗███╗   ███╗ ██████╗
  ██║████╗ ████║██╔═══██╗
  ██║██╔████╔██║██║   ██║
  ██║██║╚██╔╝██║██║   ██║
  ██║██║ ╚═╝ ██║╚██████╔╝
  ╚═╝╚═╝     ╚═╝ ╚═════╝
  Decentralized AI Training Protocol
```

> **Train any model. Contribute any data. Earn proportional rewards.**
>
> IMO is a permissionless protocol for collectively training AI models — LLMs, diffusion models, video generators, speech synthesizers, anything — across a swarm of heterogeneous consumer GPUs. No data center. No gatekeeper. Just math and incentives.

---

## How It Works

```
┌─────────────────────────────────────────────────────────────────────┐
│                        IMO PROTOCOL                                 │
│                                                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│  │ 1.PROPOSE│───>│ 2.FUND   │───>│ 3.VOTE   │───>│ 4.TRAIN  │      │
│  │  Paper   │    │ Datasets │    │ Community│    │  Swarm   │      │
│  └──────────┘    └──────────┘    └──────────┘    └────┬─────┘      │
│                                                       │             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐        │             │
│  │ 7.REWARD │<───│ 6.EVAL   │<───│ 5.SUBMIT │<───────┘             │
│  │  $IMO    │    │ Quality  │    │  Model   │                      │
│  └──────────┘    └──────────┘    └──────────┘                      │
└─────────────────────────────────────────────────────────────────────┘
```

**1. Propose** — A researcher publishes a paper describing a model architecture. No staking required.

**2. Fund with Data** — Anyone contributes datasets to the project. Datasets are quality-linted, security-scanned, deduplicated, and aggregated into a unified training corpus. The more data you contribute (quantity × quality × uniqueness), the larger your reward share.

**3. Vote** — Token holders vote to approve training. Reputation-weighted stakes determine quorum.

**4. Train** — Compute nodes join the swarm via [Hivemind](https://github.com/learning-at-home/hivemind) DHT. Gradients are compressed (Top-K / SignSGD), Byzantine-robustly aggregated (trimmed mean / Krum / coordinate-wise median), and collaboratively averaged across the swarm. Nodes with 8 GB VRAM and 800 GB VRAM participate in the same run.

**5. Submit** — The trained model is uploaded to IPFS with full provenance.

**6. Evaluate** — Community benchmarks the model. Quality score = 40% benchmarks + 25% community rating + 25% SOTA comparison + 5% code quality + 5% docs.

**7. Reward** — Quality score maps to a multiplier (0x–2x). Adjusted reward pool splits:

| Pool | Share | Recipients |
|------|-------|------------|
| Compute | 50% | GPU nodes, proportional to `time × Δloss × VRAM_ratio` |
| Data | 40% | Dataset contributors, proportional to `samples × quality × uniqueness` |
| Paper | 10% | Authors of the original research paper |

Poor models (score < 50) get **zero** rewards. Breakthrough models (score ≥ 95) get **2x** the base pool. This is by design: IMO incentivizes frontier-quality open-source models, not reward farming.

---

## Supported Model Types

IMO supports **30+ model categories** across every major modality:

| Modality | Categories |
|----------|------------|
| **Language** | LLM (pretraining), Chat/Instruct, Code |
| **Vision** | Classification, Detection, Segmentation, Generation (Stable Diffusion, etc.) |
| **Audio** | Speech Recognition, Text-to-Speech, Classification, Sound Generation |
| **Video** | Classification, Understanding, Generation (Sora-class) |
| **Multimodal** | Vision-Language, Audio-Language, Video-Language |
| **Embedding** | Text, Image, Multimodal (CLIP-class) |
| **Other** | Time-series Forecasting, Recommendation |

Each category has specific dataset requirements (data types, minimum samples, accepted formats) enforced by the data linter.

### Training Modes

| Mode | Description |
|------|-------------|
| `from_scratch` | Train a new architecture from random init |
| `full_fine_tune` | Full fine-tuning of a pretrained model |
| `lora` | Low-Rank Adaptation |
| `qlora` | Quantized LoRA |

### Diffusion Model Support

Built-in denoising diffusion training loop with:
- Cosine / linear / scaled-linear noise schedules
- Epsilon, v-prediction, and sample prediction types
- Conditional generation (text prompts, class labels)
- Works for image, video, and audio generation

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    CLI  (imo)                             │
├──────────┬──────────┬──────────────┬────────────────────┤
│   Data   │ Protocol │   Training   │       Node         │
│  Layer   │  Layer   │    Engine    │      Layer         │
├──────────┼──────────┼──────────────┼────────────────────┤
│ Linter   │ Project  │ Hivemind     │ DHT Discovery      │
│ Security │ Voting   │ Optimizer    │ VRAM Scheduler     │
│ Privacy  │ Rewards  │ Pipeline     │ Gradient           │
│ Provnce  │ Registry │ Checkpoint   │ Compression        │
│ Aggregtr │ IMO      │ Diffusion    │ Communicator       │
├──────────┴──────────┼──────────────┼────────────────────┤
│   Smart Contracts   │   PyTorch    │    Hivemind DHT    │
│ (IMOToken, Gov)     │ + HF         │                    │
└─────────────────────┴──────────────┴────────────────────┘
```

### Layer 1 — Node (`src/imo/node/`)
Peer-to-peer networking. Discovers peers via Hivemind DHT, schedules model layers across heterogeneous VRAM, compresses gradients for bandwidth efficiency.

### Layer 2 — Training Engine (`src/imo/training/`)
`DistributedTrainingEngine` orchestrates the full loop: hivemind optimizer setup, forward/backward, gradient compression/averaging, Byzantine-robust aggregation, poisoning detection, checkpointing. Supports both standard autoregressive and diffusion training objectives.

### Layer 3 — Protocol (`src/imo/protocol/`)
`Project` is the central coordination unit. Lifecycle: draft → open_for_data → voting → approved → training → evaluating → completed. Tracks dataset contributions, compute contributions, and paper authorship. Calculates quality-adjusted rewards.

### Layer 4 — Data (`src/imo/data/`)
Quality linting, code injection scanning, differential privacy (Gaussian/Laplace noise via Opacus), data provenance tracking, and multi-source dataset aggregation with proportional/balanced sampling.

### Smart Contracts (`contracts/`)
- **IMOToken.sol** — ERC-20 with quality-based reward distribution. Tracks contributor scores on-chain, proportional reward claims per pool (data/compute/paper).
- **IMOGovernance.sol** — Proposal lifecycle and stake-weighted voting with quorum enforcement.

---

## Quick Start

### Install

```bash
pip install -e ".[dev]"
```

### Create a Training Project

```bash
# Create a project to train a 7B LLM from scratch
imo project create \
  --title "Open-7B" \
  --arch llama \
  --category llm \
  --mode from_scratch \
  --max-steps 500000

# Or a diffusion image generator
imo project create \
  --title "CommunityDiffusion" \
  --arch dit \
  --category vision_generation \
  --mode from_scratch
```

### Contribute a Dataset

```bash
# Lint your dataset first
imo data lint my_dataset.parquet --min-score 0.8

# Contribute to a project
imo project contribute <project-id> my_dataset.parquet --license apache-2.0
```

### Join Training

```bash
# Start/join distributed training
imo train start <project-id> \
  --peers /ip4/203.0.113.1/tcp/12345/p2p/QmPeer1 \
  --batch-size 32 \
  --lr 1e-4
```

### Check Node Status

```bash
imo node status
```

### List Model Categories

```bash
imo categories
```

### Python API

```python
from imo.protocol.project import Project, ProjectSpec, TrainingMode, DatasetContribution

# Create a project
spec = ProjectSpec(
    model_architecture="llama",
    model_category="llm",
    training_mode=TrainingMode.FROM_SCRATCH,
    max_steps=500_000,
)
project = Project(
    id="open-7b",
    title="Open-7B",
    description="Community-trained 7B parameter LLM",
    proposer_id="researcher-alice",
    spec=spec,
)

# Open for data contributions
project.open_for_data()

# Contributors submit datasets
project.contribute_dataset(DatasetContribution(
    id="ds-001",
    contributor_id="bob",
    project_id="open-7b",
    name="web-corpus-en",
    num_samples=10_000_000,
    size_mb=50_000,
    quality_score=0.92,
    ipfs_hash="Qm...",
    license="apache-2.0",
    data_types=["text"],
    format="parquet",
))

print(f"Total samples: {project.total_samples:,}")
print(f"Contributors: {project.num_data_contributors}")
```

---

## Tokenomics

| Allocation | Share | Vesting |
|-----------|-------|---------|
| Community Rewards | 40% | Distributed per-IMO based on quality |
| Treasury | 20% | Protocol development |
| Team | 15% | 4-year linear vesting |
| Investors | 15% | 6-month lockup |
| Ecosystem | 10% | Grants, partnerships |

**Total Supply: 1,000,000,000 $IMO**

### Quality Multipliers

| Score | Level | Multiplier | Effect |
|-------|-------|-----------|--------|
| 95–100 | Breakthrough | 2.0x | Double the base pool |
| 85–94 | Excellent | 1.5x | |
| 70–84 | Good | 1.0x | Base pool |
| 50–69 | Fair | 0.5x | |
| < 50 | Poor | 0.0x | No rewards |

---

## Development

```bash
pip install -e ".[dev]"

ruff check .                                     # lint
mypy src/imo/                                    # type check
pytest                                           # all tests
pytest tests/test_data/test_linter.py -v         # single file
pytest --cov=imo --cov-report=term-missing       # coverage
```

### Key Dependencies

| Package | Purpose |
|---------|---------|
| `hivemind` | Decentralized training, DHT, gradient averaging |
| `torch` | Deep learning framework |
| `transformers` | Model architectures and tokenizers |
| `datasets` | Dataset loading and processing |
| `opacus` | Differential privacy for PyTorch |
| `web3` | Ethereum smart contract interaction |
| `pydantic` | Data validation |
| `click` | CLI framework |

---

## Security

- **Gradient Poisoning Detection** — Z-score anomaly detection + reputation tracking. Malicious nodes get exponentially decayed trust.
- **Byzantine-Robust Aggregation** — Trimmed mean, Krum, and coordinate-wise median resist up to `f` Byzantine workers.
- **Dataset Security Scanning** — AST-level code injection detection across Python, JavaScript, and shell. Blocks `eval()`, `subprocess`, `pickle`, etc.
- **Differential Privacy** — Gaussian/Laplace noise addition with formal (ε, δ)-DP guarantees via Opacus.
- **Data Provenance** — SHA-256 content hashing and full transformation lineage for integrity verification.

---

## License

MIT
