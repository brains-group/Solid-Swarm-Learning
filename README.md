# Food Swarm - Decentralized Recommender System

This project implements a decentralized machine learning system for building a food recommendation model using Swarm Learning. The system is orchestrated by a Solidity smart contract on an Ethereum-like blockchain (Anvil) and executed by a network of Python-based client nodes to store the data via a network of Solid Pods.

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


# Decentralized Swarm Learning with Solid Pod Architecture

This project implements a privacy-preserving, decentralized Swarm Learning network for a Top-N Food Recommendation System. It bridges the incentive and coordination structures of blockchain technology with the strict data sovereignty principles of the Solid ecosystem.

## The Core Idea

In a traditional machine learning paradigm, user data is aggregated into a centralized server to train a recommendation model. This creates massive data silos and strips users of their privacy. 

While standard Federated Learning attempts to solve this by keeping data on the edge and only sharing model weights, it still suffers from multiple problems like the single point of failure, Model Inversion Attacks, and model poisoning.

This project introduces a completely decoupled architecture. It uses Solid Pods to store local edge nodes in decentralized, secure, user-controlled data vault.

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

## Local Setup

Before running the full simulation, you need to install the dependencies for both the Python clients and the Node.js Solid server.

### 1. Python Dependencies
Navigate to the `learning/data/src` directory and install the required packages from `requirements.txt`.

### 2. Node.js Dependencies
From the project's root directory (`c:\Work\Swarm`), install the Node.js packages. This is required to run the Solid Pod server simulation.

npm install

### 3. Forge Dependencies
This project works by creating a local blockchain with fake ETH through Forge. This is requirered to set up the Swarm structure.

forge init swarm_orchestrator
cd swarm_orchestrator

Furthermore, the given SwarmCoordinator.sol smart contract must be moved within swarm_orchestrator/src/ to interact with the blockchain 

---

## Quick Start & Execution

To simulate the complete decentralized network locally, you will need to open five separate terminal windows to orchestrate the different components of the stack.



### Terminal 1: Boot Solid

cd solid_backend
rm -rf my-solid-data .internal
npm start


be sure to remove stuff in my-solid-data so that old profiles don't mess with client generation in the future

### Terminal 2: Generate Solid Clients and Pods

generate the specified number of solid clients through the js script:

cd solid_backend/scripts

node create_accounts.js

results are saved within user_accounts.json within secret, which stores log-in info

### Terminal 3: Boot the Blockchain
Start the local Ethereum testnet. This will spin up your local ledger and generate test wallets.

anvil

### Terminal 4: Deploy the Coordinator Contract

Deploy the Smart Contract that will orchestrate the swarm leader elections and epoch transitions.

cd swarm_orchestrator
forge script script/Deploy_SC.sol --rpc-url http://127.0.0.1:8545 --broadcast

### Terminal 5: Launch the Swarm Monitor

Start the live dashboard to monitor node registrations, epoch progress, and leader elections in real-time.

cd learning/data/src
python3 monitor_swarm.py

### Terminal 6: The Edge Nodes (Partition, Train, Evaluate)

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



Evaluation Metrics:

The network reports two primary categories of metrics to prove its efficacy:

    Classification: Accuracy, Precision, Recall, and F1-Score (Threshold = 0.5) to test general positive/negative prediction capabilities.

    Recommendation (Ranking): Hit Ratio (HR@10) and Normalized Discounted Cumulative Gain (NDCG@10) using a Leave-One-Out methodology mixed with 99 un-interacted negative samples.

### Terminal 7: Edit Edge privacy

This terminal is where you can control each pod's vulnerability and DP status via the toggle scripts, telling the blockchain whether to include or exclude a user's vulnerable data/altering the amount of differential privacy involved for that node. This code can also be run during the swarming process as well

cd learning/data/src
python3 toggle_privacy.py pod_number include/exclude
python3 toggle_dp.py pod_number dp_value

---






TODO: 
1) First make a swarm learning Agent/Class that represents a client, with logic to run local learning
    figure out how to send & receive messages for both leader election and embeddings
2) Maybe: use asincio so that the 50+ processes aren't computing for resources in the simulation
