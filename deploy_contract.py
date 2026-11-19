import json
import os
from pathlib import Path

from dotenv import load_dotenv
from solcx import compile_standard, install_solc
from web3 import Web3


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


def main():
    provider = os.getenv("WEB3_PROVIDER", "http://127.0.0.1:7545")
    account_index = int(os.getenv("BLOCKCHAIN_ACCOUNT_INDEX", "0"))
    w3 = Web3(Web3.HTTPProvider(provider))
    if not w3.is_connected():
        raise RuntimeError(f"Cannot connect to Ganache at {provider}")

    source_path = ROOT / "contracts" / "EvidenceAnchor.sol"
    source = source_path.read_text(encoding="utf-8")
    version = "0.8.24"
    install_solc(version)
    compiled = compile_standard(
        {
            "language": "Solidity",
            "sources": {source_path.name: {"content": source}},
            "settings": {
    "evmVersion": "paris",
    "optimizer": {
        "enabled": True,
        "runs": 200
    },
    "outputSelection": {
        "*": {"*": ["abi", "evm.bytecode.object"]}
    }
},
        },
        solc_version=version,
    )

    contract_data = compiled["contracts"][source_path.name]["EvidenceAnchor"]
    abi = contract_data["abi"]
    bytecode = contract_data["evm"]["bytecode"]["object"]
    account = w3.eth.accounts[account_index]
    factory = w3.eth.contract(abi=abi, bytecode=bytecode)
    tx_hash = factory.constructor().transact({"from": account})
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

    artifact_path = ROOT / os.getenv(
        "CONTRACT_ARTIFACT", "build/EvidenceAnchor.json"
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "contractName": "EvidenceAnchor",
                "address": receipt.contractAddress,
                "abi": abi,
                "deploymentTransaction": tx_hash.hex(),
                "chainId": w3.eth.chain_id,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Contract deployed: {receipt.contractAddress}")
    print(f"Chain ID: {w3.eth.chain_id}")
    print(f"Artifact saved: {artifact_path}")
    print("Copy the address into CONTRACT_ADDRESS in .env if desired.")


if __name__ == "__main__":
    main()

