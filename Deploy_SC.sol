// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "forge-std/Script.sol";
import "../src/SwarmCoordinator.sol";

contract DeployScript is Script {
    function run() external {
        // Anvil's default Account #0 private key
        uint256 deployerPrivateKey = 0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80;
        
        vm.startBroadcast(deployerPrivateKey);
        
        // Pass the epoch duration (e.g., 604800 seconds = 7 days) into the constructor
        SwarmCoordinator coordinator = new SwarmCoordinator(604800);
        
        vm.stopBroadcast();
    }
}