# Food Swarm - Decentralized Recommender System

This project implements a decentralized machine learning system for building a food recommendation model using Swarm Learning. The system is orchestrated by a Solidity smart contract on an Ethereum-like blockchain (Anvil) and executed by a network of Python-based client nodes.

## Project Structure

```
food_swarm_project/
├── swarm_orchestrator/         # Foundry smart contract project
│   ├── src/
│   │   └── SwarmCoordinator.sol
│   └── ... (Foundry config, tests, etc.)
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
```

## Key Components

### Smart Contract (`swarm_orchestrator/`)
*   **`SwarmCoordinator.sol`**: The core smart contract that manages swarm registration, epoch tracking, weight submission, leader election, and global model updates.



# Decentralized Swarm Learning with Solid Pod Architecture

This project implements a privacy-preserving, decentralized Swarm Learning network for a Top-N Food Recommendation System. It bridges the incentive and coordination structures of blockchain technology with the strict data sovereignty principles of the Solid ecosystem.

## 💡 The Core Idea: Why Combine Swarm Learning and Solid?

In a traditional machine learning paradigm, user data is aggregated into a centralized server to train a recommendation model. This creates massive data honeypots and strips users of their privacy. 

While standard Federated Learning attempts to solve this by keeping data on the edge and only sharing model weights, it still suffers from **Model Inversion Attacks**. If a traditional model shares a "User Embedding Matrix" with the network, adversaries can reverse-engineer those weights to discover exactly what items a specific user interacted with.

**The Solution:** This project introduces a completely decoupled architecture. It treats the local edge node as a **Solid Pod**—a secure, user-controlled data vault. 
1. The Swarm network globally trains and aggregates the mathematical relationships between *recipes* (The Public Recipe Embedding).
2. The user's specific tastes (The Private User Embedding) never leave the Solid Pod. 

## ⚙️ The Process: How the Network Learns

The lifecycle of this decentralized network operates in distinct phases:

### 1. Data Partitioning & Registration
The global dataset is partitioned into isolated client directories, simulating individual Solid Pods. Each Pod is assigned a cryptographic wallet, funded with test ETH, and registered to the Swarm via an Ethereum Smart Contract.

### 2. Local Training with Negative Sampling
Inside each Pod, the local PyTorch model trains on the user's explicit preferences. To teach the model how to rank (rather than just guess "Yes" to everything), the script utilizes **Strict Negative Sampling**. It generates "fake" unseen recipes for the model to evaluate, using a "Forbidden List" to ensure test data is never leaked into the training loop.

### 3. The Privacy Split & Weight Submission
When an epoch finishes, the model physically splits its brain. The private user embedding is stripped out and saved securely in the Pod. Only the public recipe weights are packaged and submitted to the blockchain. The Smart Contract enforces **Strict Epoch Checking** to ensure late nodes do not corrupt future epochs.

### 4. Leader Election & Global Aggregation
Once the Smart Contract detects that an **80% consensus threshold** has been reached, it dynamically elects a Leader node. The Leader downloads the public weights from the fastest 80%, performs Federated Averaging (FedAvg), and submits the new Global Model back to the blockchain. The nodes then download this global intelligence, merge it with their private taste vectors (`strict=False`), and begin the next epoch.

### 5. Edge Evaluation
Because evaluation cannot happen centrally without compromising privacy, it is performed entirely on the edge. Each Pod wakes up, loads the latest global model and its private state, and calculates its own ranking metrics using a Leave-One-Out methodology.

---

## 🛠️ Prerequisites

* **Python 3.8+** (Required libraries: `torch`, `pandas`, `numpy`, `web3`, `scikit-learn`)
* **Foundry** (Provides `anvil` for the local blockchain and `forge` for smart contract deployment)

---

## 🚀 Quick Start & Execution

To simulate the complete decentralized network locally, you will need to open four separate terminal windows to orchestrate the different components of the stack.

### Terminal 1: Boot the Blockchain
Start the local Ethereum testnet. This will spin up your local ledger and generate test wallets.

anvil

### Terminal 2: Deploy the Coordinator Contract

Deploy the Smart Contract that will orchestrate the swarm leader elections and epoch transitions.

cd swarm_orchestrator
forge script script/Deploy_SC.sol --rpc-url http://127.0.0.1:8545 --broadcast

### Terminal 3: Launch the Swarm Monitor

Start the live dashboard to monitor node registrations, epoch progress, and leader elections in real-time.

cd learning/data/src
python3 monitor_swarm.py

### Terminal 4: The Edge Nodes (Partition, Train, Evaluate)

This terminal simulates the actions of the edge nodes (the clients/Pods). Run these commands in sequence:

1. Partition the Data (First run only): Splits the global dataset into isolated client pods based on taste profiles.

cd learning/data/src
python3 partition_data.py

2. Register the Clients: Generates local wallets for each pod and registers them to the Swarm via the smart contract.

python3 register_clients.py

3. Run the Swarm: Kicks off the asynchronous federated training loop. Nodes will train locally, submit public weights, and wait for the leader to aggregate the global model.

python3 run_swarm.py

4. Evaluate the Network: Triggers the edge nodes to evaluate their personalized models using their private Solid Pod data, reporting final metrics.

python3 evaluate_swarm.py

📊 Evaluation Metrics

The network reports two primary categories of metrics to prove its efficacy:

    Classification: Accuracy, Precision, Recall, and F1-Score (Threshold = 0.5) to test general positive/negative prediction capabilities.

    Recommendation (Ranking): Hit Ratio (HR@10) and Normalized Discounted Cumulative Gain (NDCG@10) using a Leave-One-Out methodology mixed with 99 un-interacted negative samples.

### Terminal 5: The Vulnerable Edge Data

This terminal is where you can control each pod's vulnerability status via toggle_privacy.py, telling the blockchain whether to include or exclude a user's vulnerable data. This code can also be run during the swarming process as well

cd learning/data/src
python3 toggle_privacy.py pod_number include/exclude

---






TODO: 
1) First make a swarm learning Agent/Class that represents a client, with logic to run local learning
    figure out how to send & receive messages for both leader election and embeddings
2) Maybe: use asincio so that the 50+ processes aren't computing for resources in the simulation
