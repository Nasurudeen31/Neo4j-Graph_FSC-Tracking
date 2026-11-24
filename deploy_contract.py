import json
import os
from pathlib import Path

from dotenv import load_dotenv
from solcx import compile_standard, get_installed_solc_versions, install_solc
from web3 import Web3


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

SOLC_VERSION = "0.8.24"


def ensure_solidity_compiler():
    """Use the local compiler when available; download it only if missing."""
    installed = {str(version) for version in get_installed_solc_versions()}

    if SOLC_VERSION in installed:
        print(f"Using installed Solidity compiler {SOLC_VERSION}.")
        return

    print(f"Solidity compiler {SOLC_VERSION} is not installed.")
    print("Attempting a one-time download...")
    try:
        install_solc(SOLC_VERSION)
    except Exception as exc:
        raise RuntimeError(
            "Could not download Solidity compiler 0.8.24. "
            "Install it manually under "
            "%USERPROFILE%\\.solcx\\solc-v0.8.24\\solc.exe, then run this file again."
        ) from exc


def main():
    ensure_solidity_compiler()

    provider = os.getenv("WEB3_PROVIDER", "http://127.0.0.1:7545")
    account_index = int(os.getenv("BLOCKCHAIN_ACCOUNT_INDEX", "0"))

    web3 = Web3(Web3.HTTPProvider(provider))
    if not web3.is_connected():
        raise RuntimeError(
            f"Cannot connect to Ganache at {provider}. "
            "Start Ganache and confirm its RPC port is 7545."
        )

    accounts = web3.eth.accounts
    if not accounts:
        raise RuntimeError("Ganache returned no unlocked accounts.")
    if account_index >= len(accounts):
        raise RuntimeError(
            f"Ganache account index {account_index} does not exist. "
            f"Available accounts: {len(accounts)}."
        )

    source_path = ROOT / "contracts" / "EvidenceAnchor.sol"
    if not source_path.exists():
        raise FileNotFoundError(f"Smart contract not found: {source_path}")

    source = source_path.read_text(encoding="utf-8")
    compiled = compile_standard(
        {
            "language": "Solidity",
            "sources": {source_path.name: {"content": source}},
            "settings": {
                "evmVersion": "paris",
                "outputSelection": {
                    "*": {"*": ["abi", "evm.bytecode.object"]}
                }
            }
        },
        solc_version=SOLC_VERSION,
    )

    contract_data = compiled["contracts"][source_path.name]["EvidenceAnchor"]
    abi = contract_data["abi"]
    bytecode = contract_data["evm"]["bytecode"]["object"]

    deployment_account = accounts[account_index]
    contract_factory = web3.eth.contract(abi=abi, bytecode=bytecode)
    transaction_hash = contract_factory.constructor().transact(
        {"from": deployment_account}
    )
    receipt = web3.eth.wait_for_transaction_receipt(transaction_hash)

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
                "deploymentTransaction": transaction_hash.hex(),
                "deploymentBlock": receipt.blockNumber,
                "chainId": web3.eth.chain_id,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Contract deployment successful.")
    print(f"Contract address: {receipt.contractAddress}")
    print(f"Transaction: {transaction_hash.hex()}")
    print(f"Block number: {receipt.blockNumber}")
    print(f"Chain ID: {web3.eth.chain_id}")
    print(f"Artifact saved: {artifact_path}")


if __name__ == "__main__":
    main()

