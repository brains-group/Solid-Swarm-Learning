// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

contract SwarmCoordinator {
    
    // Represents a node's participation in a specific swarm for an epoch
    struct NodeSubmission {
        bool hasSubmitted;
        string weightsUri;
    }

    // Represents the state of an individual swarm
    struct Swarm {
        address[] registeredNodes;
        address[] submittedNodesThisEpoch;
        uint256 nodesSubmittedCount;
        uint256 currentEpoch;
        address currentLeader;
        string globalModelUri;
        uint256 epochStartTime;
        mapping(address => NodeSubmission) submissions;
    }

    // General registration status of a node address
    mapping(address => bool) public isNodeRegistered;

    // NEW: On-chain privacy flag. If true, the node will exclude its vulnerable data.
    mapping(address => bool) public excludeVulnerableData;

    // NEW: Event to log when a user changes their privacy preference
    event PrivacyPreferenceUpdated(address indexed nodeAddress, bool isExcluded);

    // Mapping from Swarm ID to the Swarm's state
    mapping(uint256 => Swarm) public swarms;

    uint256 public epochDuration;

    // Events
    event NodeRegistered(address indexed nodeAddress, uint256[] swarmIds);
    event WeightsSubmitted(address indexed nodeAddress, uint256 indexed swarmId, uint256 epoch, string uri);
    event LeaderElected(address indexed leader, uint256 indexed swarmId, uint256 epoch);
    event GlobalModelUpdated(uint256 indexed swarmId, uint256 epoch, string uri);
    event NodeKicked(address indexed nodeAddress, uint256 indexed swarmId);

    constructor(uint256 _epochDurationInSeconds) {
        // e.g., 7 days = 7 * 24 * 60 * 60 = 604800 seconds
        epochDuration = _epochDurationInSeconds;
    }

    // 1. Edge nodes call this once to join one or more swarms based on their taste profile
    function registerNode(uint256[] calldata _swarmIds) external {
        require(!isNodeRegistered[msg.sender], "Node already registered");
        require(_swarmIds.length > 0, "Must join at least one swarm");

        isNodeRegistered[msg.sender] = true;

        for (uint i = 0; i < _swarmIds.length; i++) {
            uint256 swarmId = _swarmIds[i];
            swarms[swarmId].registeredNodes.push(msg.sender);
            // If this is the first node in a new swarm, start the epoch timer
            if (swarms[swarmId].epochStartTime == 0) {
                swarms[swarmId].epochStartTime = block.timestamp;
            }
        }
        
        emit NodeRegistered(msg.sender, _swarmIds);
    }

    // NEW: Function for a node to toggle its vulnerable data sharing on or off
    function setPrivacyPreference(bool _exclude) external {
        require(isNodeRegistered[msg.sender], "Not a registered node");
        excludeVulnerableData[msg.sender] = _exclude;
        emit PrivacyPreferenceUpdated(msg.sender, _exclude);
    }

    // 2. Nodes call this for each swarm they are a part of when they finish local training
    // ADDED: _epoch parameter to prevent late submissions
    function submitWeights(uint256 _swarmId, uint256 _epoch, string calldata _weightsUri) external {
        require(isNodeRegistered[msg.sender], "Not a registered node");
        
        Swarm storage swarm = swarms[_swarmId];

        // --- STRICT EPOCH CHECK ---
        require(_epoch == swarm.currentEpoch, "Epoch mismatch: Node is too late for this epoch");

        NodeSubmission storage submission = swarm.submissions[msg.sender];
        require(!submission.hasSubmitted, "Already submitted for this epoch in this swarm");

        submission.weightsUri = _weightsUri;
        submission.hasSubmitted = true;
        swarm.nodesSubmittedCount++;
        swarm.submittedNodesThisEpoch.push(msg.sender);

        emit WeightsSubmitted(msg.sender, _swarmId, swarm.currentEpoch, _weightsUri);

        // Check if all registered nodes in this swarm have submitted their weights
        _checkAndElectLeader(_swarmId);
    }

    // 3. Internal logic to elect a leader for a swarm once 80% of members are done
    function _checkAndElectLeader(uint256 _swarmId) internal {
        Swarm storage swarm = swarms[_swarmId];
        
        if (swarm.registeredNodes.length == 0) {
            return;
        }

        // Calculate the 80% threshold
        uint256 threshold = (swarm.registeredNodes.length * 80) / 100;

        // Check if we hit the 80% threshold
        if (swarm.nodesSubmittedCount < threshold) {
            return; // Still waiting to reach 80%, exit the function
        }
        
        // CRITICAL: Prevent re-electing a leader if the remaining 20% of nodes submit late
        if (swarm.currentLeader != address(0)) {
            return;
        }

        // Threshold reached! Elect a leader.
        uint256 leaderIndex = swarm.currentEpoch % swarm.submittedNodesThisEpoch.length;
        swarm.currentLeader = swarm.submittedNodesThisEpoch[leaderIndex];

        emit LeaderElected(swarm.currentLeader, _swarmId, swarm.currentEpoch);
    }

    // 4. The elected leader for a swarm calls this after merging weights for that swarm
    function submitGlobalModel(uint256 _swarmId, string calldata _globalModelUri) external {
        Swarm storage swarm = swarms[_swarmId];
        require(msg.sender == swarm.currentLeader, "Only the elected leader for this swarm can submit the global model");

        swarm.globalModelUri = _globalModelUri;
        
        // NOTE: We removed _kickInactiveNodes(_swarmId) here. 
        // Since we aggregate at 80%, we don't want to punish the remaining 20% of nodes just for being a bit slow.
        
        // Reset state for the next epoch for this swarm
        for (uint i = 0; i < swarm.registeredNodes.length; i++) {
            address nodeAddress = swarm.registeredNodes[i];
            swarm.submissions[nodeAddress].hasSubmitted = false;
            swarm.submissions[nodeAddress].weightsUri = "";
        }
        
        delete swarm.submittedNodesThisEpoch;
        swarm.nodesSubmittedCount = 0;
        swarm.currentEpoch++;
        swarm.currentLeader = address(0); // Clear leader until next election
        swarm.epochStartTime = block.timestamp;

        emit GlobalModelUpdated(_swarmId, swarm.currentEpoch - 1, _globalModelUri);
    }

    // Helper function for a swarm's leader to get all weight URIs for that swarm
    function getAllWeightsUris(uint256 _swarmId) external view returns (string[] memory) {
        Swarm storage swarm = swarms[_swarmId];
        string[] memory uris = new string[](swarm.submittedNodesThisEpoch.length);
        for (uint i = 0; i < swarm.submittedNodesThisEpoch.length; i++) {
            address nodeAddress = swarm.submittedNodesThisEpoch[i];
            uris[i] = swarm.submissions[nodeAddress].weightsUri;
        }
        return uris;
    }

    function _kickInactiveNodes(uint256 _swarmId) internal {
        Swarm storage swarm = swarms[_swarmId];
        address[] memory activeNodes = swarm.submittedNodesThisEpoch;
        
        for(uint i=0; i < swarm.registeredNodes.length; i++){
            if(!swarm.submissions[swarm.registeredNodes[i]].hasSubmitted){
                emit NodeKicked(swarm.registeredNodes[i], _swarmId);
            }
        }
        swarm.registeredNodes = activeNodes;
    }
}