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
┌────────────────────────────────────────────────────────────────┐
│                          IMO PROTOCOL                          │
│                                                                │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │ 1.PROPOSE│───>│ 2.FUND   │───>│ 3.VOTE   │───>│ 4.TRAIN  │  │
│  │  Paper   │    │ Datasets │    │ Community│    │  Swarm   │  │
│  └──────────┘    └──────────┘    └──────────┘    └────┬─────┘  │
│                                                       │        │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐         │        │
│  │ 7.REWARD │<───│ 6.EVAL   │<───│ 5.SUBMIT │<────────┘        │
│  │  $IMO    │    │ Quality  │    │  Model   │                  │
│  └──────────┘    └──────────┘    └──────────┘                  │
└────────────────────────────────────────────────────────────────┘
```

**1. Propose** — A researcher publishes a paper describing a model architecture. No staking required.

**2. Fund with Data** — Anyone contributes datasets to the project. Datasets are quality-linted, security-scanned, deduplicated, and aggregated into a unified training corpus. The more data you contribute (quantity × quality × uniqueness), the larger your reward share.

**3. Vote** — Token holders vote to approve training. Reputation-weighted stakes determine quorum.

**4. Train** — Compute nodes join the swarm via [Hivemind](https://github.com/learning-at-home/hivemind) DHT. Gradients are compressed (Top-K / SignSGD), Byzantine-robustly aggregated (trimmed mean / Krum / coordinate-wise median), and collaboratively averaged across the swarm. Nodes with 8 GB VRAM and 800 GB VRAM participate in the same run.

**5. Submit** — The trained model is uploaded to IPFS with full provenance.

**6. Evaluate** — Community benchmarks the model. Quality score = 40% benchmarks + 25% community rating + 25% SOTA comparison + 5% code quality + 5% docs.

**7. Reward** — Quality score maps to a multiplier (0x–2x). Adjusted reward pool splits:

| Pool    | Share | Recipients                                                    |
|---------|-------|---------------------------------------------------------------|
| Compute | 50%   | GPU nodes, proportional to `time * delta_loss * VRAM_ratio`  |
| Data    | 40%   | Dataset contributors, proportional to `samples * quality * uniqueness` |
| Paper   | 10%   | Authors of the original research paper                        |

Poor models (score < 50) get **zero** rewards. Breakthrough models (score ≥ 95) get **2x** the base pool. This is by design: IMO incentivizes frontier-quality open-source models, not reward farming.

---

## Supported Model Types

IMO supports **30+ model categories** across every major modality:

| Modality       | Categories                                                      |
|----------------|-----------------------------------------------------------------|
| **Language**   | LLM (pretraining), Chat/Instruct, Code                         |
| **Vision**     | Classification, Detection, Segmentation, Generation (SD, etc.) |
| **Audio**      | Speech Recognition, Text-to-Speech, Classification, Generation |
| **Video**      | Classification, Understanding, Generation (Sora-class)         |
| **Multimodal** | Vision-Language, Audio-Language, Video-Language                  |
| **Embedding**  | Text, Image, Multimodal (CLIP-class)                            |
| **Other**      | Time-series Forecasting, Recommendation                         |

Each category has specific dataset requirements (data types, minimum samples, accepted formats) enforced by the data linter.

### Training Modes

| Mode               | Description                              |
|--------------------|------------------------------------------|
| `from_scratch`     | Train a new architecture from random init|
| `full_fine_tune`   | Full fine-tuning of a pretrained model   |
| `lora`             | Low-Rank Adaptation                      |
| `qlora`            | Quantized LoRA                           |
| `continual_pretrain`| Continue pretraining on new data        |
| `distillation`     | Knowledge distillation from teacher model|
| `hybrid`           | Combined approaches                      |

### Diffusion Model Support

Built-in denoising diffusion training loop with:
- Cosine / linear / scaled-linear noise schedules
- Epsilon, v-prediction, and sample prediction types
- Conditional generation (text prompts, class labels)
- Works for image, video, and audio generation

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                            CLI  (imo)                               │
│          Interactive Dashboard / Project Wizard / Rich UI           │
├────────────┬──────────┬────────────────────────────┬────────────────┤
│    Data    │ Protocol │     Training Pipeline      │     Node       │
│   Layer    │  Layer   │                            │     Layer      │
├────────────┼──────────┤  ┌──────────────────────┐  ├────────────────┤
│ Linter     │ Project  │  │      Toolkits        │  │ DHT Discovery  │
│ Security   │ Voting   │  │    (pluggable)       │  │ VRAM Scheduler │
│ Privacy    │ Rewards  │  │                      │  │ Gradient       │
│ Provenance │ Registry │  │  load_model()        │  │ Compression    │
│ Aggregator │ IMO      │  │  compute_loss()      │  │ mTLS Transport │
│ Cleanlab   │ Snapshot │  │  create_optimizer()  │  │ Node Auth      │
│ Semgrep    │          │  └───────────┬──────────┘  │ Node Manager   │
│            │          │              │             │                │
│            │          │  ┌───────────▼──────────┐  │                │
│            │          │  │   Training Engine    │  │                │
│            │          │  │                      │  │                │
│            │          │  │   Preflight Gate     │◄─┤                │
│            │          │  │   Hivemind DHT       │  │                │
│            │          │  │   Pipeline Parallel  │  │                │
│            │          │  │   Byzantine Aggreg   │  │                │
│            │          │  │   FLTrust / Verifier │  │                │
│            │          │  │   Canary Detection   │  │                │
│            │          │  │   Checkpointing      │  │                │
│            │          │  └──────────────────────┘  │                │
├────────────┴──────────┼────────────────────────────┼────────────────┤
│   Smart Contracts     │          PyTorch           │  Hivemind DHT  │
│ (IMOToken, Governance)│       + HF Ecosystem       │                │
└───────────────────────┴────────────────────────────┴────────────────┘
```

### Training Pipeline — How Toolkits and Engine Work Together

The core design insight: **Toolkits know how to load models and compute loss. The Engine knows how to distribute training across a GPU swarm.** They're separated so that any training backend (HF Trainer, Unsloth, Diffusers, etc.) can run on a decentralized Hivemind network without modification.

```
┌─────────────────────────────────────────────────────────────────┐
│ Toolkit (pluggable)           Engine (distributed)              │
│                                                                 │
│ 1. load_model()          ──►  Receives nn.Module                │
│    - HF pretrained             - Data parallel: replicate model │
│    - LoRA/QLoRA wrap           - Pipeline parallel: split layers│
│    - From-scratch init           across nodes via BlockServer   │
│                                                                 │
│ 2. compute_loss()        ──►  Called inside training loop       │
│    - Causal LM loss            - Gradients compressed (Top-K)   │
│    - Diffusion denoising       - All-reduce via Hivemind        │
│    - Distillation KL           - Byzantine aggregation          │
│                                                                 │
│ 3. create_optimizer()    ──►  Wrapped by Hivemind.Optimizer     │
│    - AdamW, SGD, fused         - Decentralized step syncing     │
│                                                                 │
│ 4. post_training()       ◄──  Called after training completes   │
│    - Merge LoRA weights        - Final checkpoint saved         │
│    - Export safetensors        - Contributions recorded         │
└─────────────────────────────────────────────────────────────────┘
```

### Parallelism Modes

| Mode                        | Mechanism                                      | When to Use                        |
|-----------------------------|-------------------------------------------------|------------------------------------|
| **Data Parallel** (default) | Full model on each node; gradients compressed   | Model fits on one GPU              |
|                             | (Top-K) and averaged via Hivemind               | (most LoRA, small-to-mid models)   |
| **Pipeline Parallel**       | Model split by layers across nodes via          | Model too large for one GPU        |
|                             | `BlockServer`/`RemoteSequential`; DHT routing   | (70B+ params)                      |
|                             | with fault-tolerant rerouting                   |                                    |

### Built-in Toolkits (`src/imo/toolkits/`)

| Toolkit              | Best For                          | Modes                              | Min VRAM |
|----------------------|-----------------------------------|------------------------------------|----------|
| **HF Trainer** (default) | Universal LLM/classification  | all 7 modes                        | 8 GB     |
| **Unsloth**          | Fast LoRA/QLoRA, 70% less memory  | LoRA, QLoRA, fine-tune             | 6 GB     |
| **Axolotl**          | YAML-driven, DPO/RLHF built-in   | all 7 modes                        | 8 GB     |
| **Diffusers**        | SD/SDXL/Flux diffusion            | from_scratch, fine-tune, LoRA      | 8 GB     |
| **Musubi-Tuner**     | Wan2.1 video generation           | fine-tune, LoRA                    | 12 GB    |
| **AI-Toolkit**       | Flux/SD image LoRA/DreamBooth     | fine-tune, LoRA                    | 12 GB    |

All toolkits plug into the same `DistributedTrainingEngine`. The engine handles Hivemind DHT, layer splitting, gradient aggregation, and Byzantine fault tolerance — regardless of which toolkit loaded the model.

### Node Layer (`src/imo/node/`)
Peer-to-peer networking with full identity and encryption. Discovers peers via Hivemind DHT, schedules model layers across heterogeneous VRAM (bin-packing), compresses gradients (Top-K sparsification, sign encoding). Every node holds an Ed25519 identity keypair and communicates over mutual TLS. `NodeManager` orchestrates the full lifecycle: recruitment → registration → challenge-response authentication → TLS channel → training → monitoring → eviction.

### Training Engine (`src/imo/training/`)
`DistributedTrainingEngine` orchestrates the full distributed loop. Accepts a `TrainingToolkit` for model loading and loss computation. Handles: mandatory pre-flight security gate, Hivemind optimizer setup, pipeline parallelism (`BlockServer` / `RemoteSequential`), gradient compression/averaging, Byzantine-robust aggregation (trimmed mean / Krum / FLTrust), poisoning detection, redundant computation verification, canary-based integrity monitoring, warmup trust for new nodes, and checkpointing. Factory method: `DistributedTrainingEngine.from_toolkit(config, toolkit, spec)`.

### Protocol Layer (`src/imo/protocol/`)
`Project` is the central coordination unit. Lifecycle: draft → open_for_data → voting → approved → training → evaluating → completed. 7 training modes: from_scratch, full_fine_tune, LoRA, QLoRA, continual_pretrain, distillation, hybrid. Tracks dataset and compute contributions. Quality-adjusted reward distribution (40% data / 50% compute / 10% paper).

### Data Layer (`src/imo/data/`)
Quality linting (including ML-powered label issue detection via Cleanlab), code injection scanning (AST-level multi-language + Semgrep), differential privacy (Gaussian/Laplace noise via Opacus), secure aggregation (pairwise additive masking), data provenance tracking (SHA-256 hashing), and multi-source dataset aggregation with proportional/balanced sampling.

### Smart Contracts (`contracts/`)
- **IMOToken.sol** — ERC-20 with quality-based reward distribution. Reentrancy-guarded contributor score tracking, proportional reward claims per pool (data/compute/paper).
- **IMOGovernance.sol** — Proposal lifecycle with balance-snapshot voting (anti flash-loan), stake-weighted quorum enforcement.

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
# Start/join distributed training (data parallel — default)
imo train start <project-id> \
  --peers /ip4/203.0.113.1/tcp/12345/p2p/QmPeer1 \
  --batch-size 32 \
  --lr 1e-4

# Pipeline parallel — split model layers across GPU nodes
# Required for models too large for a single GPU
imo train start <project-id> \
  --parallelism pipeline_parallel \
  --peers /ip4/203.0.113.1/tcp/12345/p2p/QmPeer1
```

### Manage Toolkits

```bash
# List all available training backends
imo toolkit list

# Show details and check environment readiness
imo toolkit info hf_trainer

# Install a toolkit's dependencies
imo toolkit install unsloth
```

### Security

```bash
# Run pre-training security checks on a model
imo security preflight ./model.safetensors --dataset data.parquet

# Scan a file for code injection vulnerabilities
imo security scan untrusted_script.py --scanner both

# Show all active security mechanisms
imo security audit
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

| Allocation          | Share | Vesting                                |
|---------------------|-------|----------------------------------------|
| Community Rewards   | 40%   | Distributed per-IMO based on quality   |
| Treasury            | 20%   | Protocol development                   |
| Team                | 15%   | 4-year linear vesting                  |
| Investors           | 15%   | 6-month lockup                         |
| Ecosystem           | 10%   | Grants, partnerships                   |

**Total Supply: 1,000,000,000 $IMO**

### Quality Multipliers

| Score    | Level        | Multiplier | Effect                 |
|----------|--------------|------------|------------------------|
| 95 - 100 | Breakthrough | 2.0x       | Double the base pool   |
| 85 - 94  | Excellent    | 1.5x       |                        |
| 70 - 84  | Good         | 1.0x       | Base pool              |
| 50 - 69  | Fair         | 0.5x       |                        |
| < 50     | Poor         | 0.0x       | No rewards             |

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

| Package          | Purpose                                           |
|------------------|---------------------------------------------------|
| `hivemind`       | Decentralized training, DHT, gradient averaging   |
| `torch`          | Deep learning framework                           |
| `transformers`   | Model architectures and tokenizers                |
| `datasets`       | Dataset loading and processing                    |
| `cryptography`   | Ed25519 node identity, X.509 certificates, mTLS   |
| `opacus`         | Differential privacy for PyTorch                  |
| `web3`           | Ethereum smart contract interaction               |
| `pydantic`       | Data validation                                   |
| `click`          | CLI framework                                     |
| `rich`           | Terminal UI (tables, colors, progress)             |

Optional security extras (`pip install -e ".[security]"`):

| Package          | Purpose                                           |
|------------------|---------------------------------------------------|
| `cleanlab`       | ML-powered label issue and outlier detection       |
| `semgrep`        | AST-level code security scanning                   |

---

## Security

IMO is designed for adversarial environments where any participant may be malicious. Security is enforced at every stage of the training lifecycle:

### Node Identity & Authentication

| Mechanism                      | Description                                                                 |
|-------------------------------|-----------------------------------------------------------------------------|
| **Ed25519 Identity**          | Each node holds a cryptographic keypair; `node_id = SHA-256(public_key)`    |
| **Challenge-Response Auth**   | Nodes prove identity by signing random nonces before joining training       |
| **Admission Control**         | Per-project policies: open, invite-only, or stake-required                  |
| **mTLS Transport**            | All node-to-node communication encrypted via mutual TLS 1.3                 |
| **Node Lifecycle Management** | Full lifecycle: recruit → register → authenticate → connect → train → evict |
| **Heartbeat Monitoring**      | Dead or unresponsive nodes are automatically evicted                        |

### Pre-Training Security Gate (Preflight)

Training **cannot start** until all preflight checks pass:

| Check                        | What It Does                                                               |
|------------------------------|----------------------------------------------------------------------------|
| **SafeModelLoader**          | Prefers safetensors; blocks pickle RCE in `.pt` files (`weights_only=True`)|
| **ModelIntegrityVerifier**   | SHA-256 per-parameter manifest; detects tampered weights                   |
| **ConfigValidator**          | Bounds-checks learning rate, batch size, trim ratio, etc.                  |
| **DatasetQuarantine**        | Security-scans all datasets for code injection before training begins      |
| **WarmupTrustPolicy**        | New nodes start at 0.3x trust weight, ramp to 1.0x over 100 steps         |
| **CanaryDetector**           | Embeds sentinel samples; monitors for loss spikes indicating poisoning     |

### Training Runtime Defense

| Mechanism                         | Description                                                                  |
|-----------------------------------|-----------------------------------------------------------------------------|
| **Byzantine-Robust Aggregation**  | Trimmed mean, Krum, coordinate-wise median — resists up to `f` Byzantine workers |
| **FLTrust (Trusted Root)**        | Weights node gradients by cosine similarity to a trusted reference gradient (NDSS 2021) |
| **Redundant Verification**        | Spot-checks batches across multiple nodes; flags divergent results          |
| **Gradient Anomaly Detection**    | Z-score based anomaly detection + exponential reputation decay              |
| **Poisoning Detection**           | Automatic node banning after configurable strike limit                       |
| **TLS-Enforced Gradient Exchange**| GradientCommunicator rejects peers without verified TLS channels            |

### Data Layer Security

| Mechanism                    | Description                                                                  |
|------------------------------|-----------------------------------------------------------------------------|
| **Code Injection Scanning**  | AST-level detection across Python, JavaScript, shell (blocks `eval`, `exec`, `pickle`, etc.) |
| **Semgrep Integration**      | Custom SAST rules for dangerous-exec, network-access, unsafe-deserialization |
| **Cleanlab Integration**     | ML-powered label issue detection and out-of-distribution sample flagging     |
| **Differential Privacy**     | Gaussian/Laplace noise with formal (ε, δ)-DP guarantees via Opacus          |
| **Secure Aggregation**       | Pairwise additive masking — individual gradients never exposed (Bonawitz CCS 2017) |
| **Data Provenance**          | SHA-256 content hashing and full transformation lineage                      |

### Smart Contract Security

| Mechanism                    | Description                                                                  |
|------------------------------|-----------------------------------------------------------------------------|
| **Reentrancy Guard**         | All state-changing functions use OpenZeppelin `nonReentrant`                 |
| **Checks-Effects-Interactions** | State updated before external calls in reward distribution                |
| **Voting Snapshot**          | Voter balances locked before voting to prevent flash-loan manipulation       |
| **Role-Based Access**        | `MINTER_ROLE`, `EVALUATOR_ROLE`, `TREASURY_ROLE` via OpenZeppelin AccessControl |

### Serialization Safety

| Layer         | Before                  | After                                          |
|---------------|-------------------------|------------------------------------------------|
| Transport     | `pickle.loads` (RCE)    | JSON serialization only — no arbitrary code execution |
| Checkpoints   | `torch.load` (unsafe)   | `torch.load(weights_only=True)` + safetensors preferred |
| Model Loading | Unrestricted            | SafeModelLoader scans for dangerous pickle modules      |

---

## Acknowledgments

IMO stands on the shoulders of two groundbreaking projects. Without them, decentralized swarm training would still be a whiteboard sketch.

- **[Hivemind](https://github.com/learning-at-home/hivemind)** — Decentralized deep learning framework by [Learning@home](https://github.com/learning-at-home) (Max Ryabinin, Alexander Borzunov, et al.). Hivemind provides the DHT-based peer discovery, collaborative optimizer, and decentralized gradient averaging that form the backbone of IMO's training engine. Paper: *[Towards Crowdsourced Training of Large Neural Networks using Decentralized Mixture-of-Experts](https://arxiv.org/abs/2002.04013)*.

- **[Petals](https://github.com/bigscience-workshop/petals)** — Run large language models at home, BitTorrent-style, by [BigScience](https://github.com/bigscience-workshop) (Alexander Borzunov, Max Ryabinin, et al.). IMO's pipeline parallelism architecture — BlockServer, RemoteSequential, DHT-based block routing, and fault-tolerant rerouting — is directly inspired by Petals' elegant design for splitting transformer blocks across consumer GPUs. Paper: *[Petals: Collaborative Inference and Fine-tuning of Large Models](https://arxiv.org/abs/2209.01188)*.

IMO's pluggable training toolkit system is powered by these outstanding open-source projects:

- **[Transformers](https://github.com/huggingface/transformers)** & **[Diffusers](https://github.com/huggingface/diffusers)** — by [Hugging Face](https://github.com/huggingface). The Trainer API and diffusion training pipelines serve as IMO's default backends for LLM/classification and image/video generation respectively.

- **[Unsloth](https://github.com/unslothai/unsloth)** — by [Daniel & Michael Han](https://github.com/unslothai). Dramatically faster and more memory-efficient LoRA/QLoRA fine-tuning, enabling 70% less VRAM usage on consumer GPUs.

- **[Axolotl](https://github.com/axolotl-ai-cloud/axolotl)** — by [Wing Lian](https://github.com/winglian) and the [OpenAccess AI Collective](https://github.com/axolotl-ai-cloud). Config-driven training with built-in DPO/RLHF support, covering all 7 training modes.

- **[Musubi-Tuner](https://github.com/kohya-ss/musubi-tuner)** — by [kohya-ss](https://github.com/kohya-ss). Fine-tuning toolkit for Wan2.1 video generation models.

- **[AI-Toolkit](https://github.com/ostris/ai-toolkit)** — by [Ostris](https://github.com/ostris). Flux/Stable Diffusion LoRA and DreamBooth training for image generation.

We are deeply grateful to all these teams for open-sourcing their work and making decentralized AI training a reality.

---

## License

MIT
