// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/// @title CaseAnchor
/// @notice Anchors a SHA-256 hash of a finalized fraud-detection case
///         document on-chain for tamper-evidence. Stores no case data
///         itself - only the hash (fingerprint) and its case_id.
contract CaseAnchor {

    struct Anchor {
        bytes32 caseHash;
        uint256 anchoredAt;
        address anchoredBy;
        bool exists;
    }

    // case_id (string) => anchor record
    mapping(string => Anchor) private anchors;

    event CaseAnchored(
        string indexed caseId,
        bytes32 caseHash,
        uint256 anchoredAt,
        address anchoredBy
    );

    /// @notice Anchors a case hash on-chain. Reverts if this case_id
    ///         has already been anchored (a case is finalized once).
    function anchorCase(string calldata caseId, bytes32 caseHash) external {
        require(!anchors[caseId].exists, "Case already anchored");

        anchors[caseId] = Anchor({
            caseHash: caseHash,
            anchoredAt: block.timestamp,
            anchoredBy: msg.sender,
            exists: true
        });

        emit CaseAnchored(caseId, caseHash, block.timestamp, msg.sender);
    }

    /// @notice Reads back the anchored hash for a case_id, for
    ///         tamper-evidence verification.
    function getAnchor(string calldata caseId)
        external
        view
        returns (bytes32 caseHash, uint256 anchoredAt, address anchoredBy, bool exists)
    {
        Anchor memory a = anchors[caseId];
        return (a.caseHash, a.anchoredAt, a.anchoredBy, a.exists);
    }
}
