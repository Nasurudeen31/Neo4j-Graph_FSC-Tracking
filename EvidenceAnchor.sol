// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

contract EvidenceAnchor {
    struct Anchor {
        string batchId;
        string evidenceId;
        string eventType;
        string ipfsCid;
        bytes32 contentHash;
        uint256 anchoredAt;
        address submitter;
    }

    address public immutable owner;
    mapping(bytes32 => Anchor) private anchors;

    event EvidenceAnchored(
        string indexed evidenceId,
        string indexed batchId,
        string eventType,
        string ipfsCid,
        bytes32 contentHash,
        uint256 anchoredAt,
        address indexed submitter
    );

    constructor() {
        owner = msg.sender;
    }

    function anchorEvidence(
        string calldata batchId,
        string calldata evidenceId,
        string calldata eventType,
        string calldata ipfsCid,
        bytes32 contentHash
    ) external {
        require(msg.sender == owner, "Only owner can anchor evidence");
        bytes32 key = keccak256(bytes(evidenceId));
        require(anchors[key].anchoredAt == 0, "Evidence already anchored");

        anchors[key] = Anchor({
            batchId: batchId,
            evidenceId: evidenceId,
            eventType: eventType,
            ipfsCid: ipfsCid,
            contentHash: contentHash,
            anchoredAt: block.timestamp,
            submitter: msg.sender
        });

        emit EvidenceAnchored(
            evidenceId,
            batchId,
            eventType,
            ipfsCid,
            contentHash,
            block.timestamp,
            msg.sender
        );
    }

    function getEvidence(string calldata evidenceId)
        external
        view
        returns (Anchor memory)
    {
        bytes32 key = keccak256(bytes(evidenceId));
        require(anchors[key].anchoredAt != 0, "Evidence not found");
        return anchors[key];
    }

    function verifyEvidence(
        string calldata batchId,
        string calldata evidenceId,
        string calldata eventType,
        string calldata ipfsCid,
        bytes32 contentHash
    ) external view returns (bool) {
        bytes32 key = keccak256(bytes(evidenceId));
        Anchor storage item = anchors[key];
        if (item.anchoredAt == 0) return false;
        return keccak256(bytes(item.batchId)) == keccak256(bytes(batchId))
            && keccak256(bytes(item.eventType)) == keccak256(bytes(eventType))
            && keccak256(bytes(item.ipfsCid)) == keccak256(bytes(ipfsCid))
            && item.contentHash == contentHash;
    }
}
