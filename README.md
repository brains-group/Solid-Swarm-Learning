# General Swarm - Decentralized Recommender System

This project implements a decentralized machine learning system for building a general recommendation model using Swarm Learning. In this demo we showcase it via both a food and movie dataset. The system is orchestrated by a Solidity smart contract on an Ethereum-like blockchain (Anvil) and executed by a network of Python-based client nodes to store the data via a network of Solid Pods.

## Project Structure

```
food_swarm_project/
├── swarm_orchestrator/         # Foundry smart contract project
│   ├── src/
│   │   └── SwarmCoordinator.sol # The core smart contract that manages swarm registration, epoch tracking, weight submission, leader election, vulnerability status, and global model updates.
│   └── ... (Foundry config, tests, etc.)
├── node_modules/               # npm logic 
│
└── learning/                   # Python ML & simulation environment
    └── data/
        ├── global/             # Holds the raw, global datasets
        │   ├── RAW_interactions.csv
        │   └── PP_recipes.csv
        │
        ├── pods/               # Isolated client data & model storage
        │   ├── client_1/
        │   │   ├── profile.json
        │   │   ├── train_common.csv # common data that is always included in swarm
                ├── vulnerable.csv # vulnerable data the user can choose to exclude
        │   │   └── test.csv
        │   └── ... (client_2, client_3, etc.)
        │
        └── src/                # Python source scripts for the swarm
            ├── config.py
            ├── partition_data.py
            ├── register_clients.py
            ├── run_swarm.py
            ├── train_node.py
            ├── monitor_swarm.py
            └── evaluate_swarm.py
    requirements.txt
├── solid_backend/
│   ├── src/   # keeps express server logic
│   ├── package.json                #(Node dependencies)
│   └── tsconfig.json               #(TypeScript config)
├── secret/         # holds each client authentication details
```

## The Process

The lifecycle of this decentralized network operates in distinct phases:

### 1. Data Partitioning & Registration
The global dataset, which is sourced from Food.com (https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)  is partitioned into isolated client directories, simulating individual Solid Pods. Each client is assigned a cryptographic wallet, funded with test ETH, and is registered to the Swarm via an Ethereum Smart Contract. The clients are also given control of a single Solid Pod through a private and public key token pair, which they use to store and access their data.

### 2. The Privacy Split & Weight Submission
When an epoch finishes, the private user embedding is stripped out and saved securely in the Pod, and only the public recipe weights are packaged and submitted to the blockchain. The Smart Contract enforces strict epoch checking to ensure nodes that submit their updates late do not corrupt the developed model.

### 3. Leader Election & Global Aggregation
Once the Smart Contract detects that an 80% consensus threshold has been reached, it dynamically elects a Leader node. The Leader downloads the public weights and submits the new Global Model back to the blockchain. All the pending nodes then download this global intelligence and begin the next epoch.

### 5. Edge Evaluation
Because evaluation cannot happen centrally without compromising privacy, evaluation (for now) is performed entirely on the edge. Each Pod wakes up, loads the latest global model and its private state, and calculates its own ranking metrics.

## Minimal requirements (what to install) to run this project:

- Python 3.10+ with these Python packages (see `learning/requirements.txt`):
    - numpy, pandas, torch, scikit-learn, matplotlib
- Node.js and npm (for the Solid server in `solid_backend`)
- Foundry tooling (Forge / Anvil) for local EVM testing
- tmux (optional, used by master scripts)
- the ml-32m dataset created by MovieLens (https://grouplens.org/datasets/movielens/), saved within learning/data/global in its own folder called ml-32m

See the `learning/requirements.txt` for the full Python dependency list. The repository contains orchestration scripts under the project root and Python code under `learning/data/src/`.

---

## Quick Start & Execution

To Edit Client/Round Values, please open config.py within learning/data/src/

### `master_swarm.sh`

This is the main script to start a new swarm learning experiment from scratch.

**Functionality:**
-   Starts a local Anvil blockchain instance.
-   Deploys the orchestrator smart contract.
-   Sets up and starts the Solid servers for the client nodes.
-   Starts the Python client nodes, which will then begin the decentralized training process.

**How to run:**
```bash
./master_swarm.sh
```




### `master_swarm_rerun.sh`

This script is used to rerun an experiment, assuming nothing has been changed in config.py, to skip the long data pertitioning steps

**How to run:**
```bash
./master_swarm_rerun.sh
```

### `master_swarm_fix.sh`

This script helps start back up any previous run that crashed mid-training.

**How to run:**
```bash
./master_swarm_fix.sh
```

### Other Scripts

-   `python3 learning/data/src monitor_swarm.py' can be used to keep track of current training progress
-   `python3 learning/data/src evaluate_swarm.py' can be used to observe current epoch learning direction, as well as rerun final metrics