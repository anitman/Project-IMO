// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "./IMOToken.sol";

/**
 * @title IMOGovernance
 * @dev Manages IMO proposals, voting, and training lifecycle
 */
contract IMOGovernance is AccessControl, ReentrancyGuard {
    IMOToken public immutable IMO_TOKEN;

    bytes32 public constant PROPOSER_ROLE = keccak256("PROPOSER_ROLE");
    bytes32 public constant VOTER_ROLE = keccak256("VOTER_ROLE");

    // Voting parameters
    uint256 public constant VOTING_PERIOD = 7 days;
    uint256 public constant MIN_STAKE = 1000 * 10 ** 18;  // 1000 IMO
    uint256 public constant QUORUM_THRESHOLD = 10000 * 10 ** 18;  // 10000 IMO

    enum IMOStatus { Submitted, Voting, Approved, Training, Completed, Failed }
    enum TrainingMode { NewArchitecture, FineTuning }

    struct TrainingSpec {
        string modelArchitecture;
        string modelCategory;  // e.g., "llm", "multimodal_vlm"
        string[] datasetIds;
        uint256 maxSteps;
        TrainingMode mode;
    }

    struct IMO {
        uint256 id;
        address proposer;
        string title;
        string abstract;
        string ipfsHash;
        TrainingSpec trainingSpec;
        IMOStatus status;
        uint256 votingDeadline;
        uint256 totalStake;
        uint256 quorumRequired;
        mapping(address => bool) hasVoted;
        mapping(address => bool) votes;  // true = yes, false = no
    }

    mapping(uint256 => IMO) internal imos;
    uint256 public imoCount;
    mapping(uint256 => address[]) public imoVoters;

    // Snapshot: lock voter's balance at voting start to prevent flash-loan manipulation
    mapping(uint256 => mapping(address => uint256)) public votingSnapshot;

    event IMOProposed(
        uint256 indexed imoId,
        address indexed proposer,
        string title,
        string modelCategory
    );

    event VoteCast(
        uint256 indexed imoId,
        address indexed voter,
        bool support,
        uint256 stake
    );

    event IMOApproved(uint256 indexed imoId);
    event TrainingStarted(uint256 indexed imoId);
    event TrainingCompleted(uint256 indexed imoId);

    constructor(IMOToken _token) {
        IMO_TOKEN = _token;
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(PROPOSER_ROLE, msg.sender);
        _grantRole(VOTER_ROLE, msg.sender);
    }

    /**
     * @dev Submit a new IMO proposal
     */
    function proposeIMO(
        string memory title,
        string memory abstract,
        string memory ipfsHash,
        string memory modelArchitecture,
        string memory modelCategory,
        string[] memory datasetIds,
        uint256 maxSteps,
        TrainingMode mode
    ) external onlyRole(PROPOSER_ROLE) nonReentrant {
        uint256 imoId = imoCount;
        IMO storage newIMO = imos[imoId];
        newIMO.id = imoId;
        newIMO.proposer = msg.sender;
        newIMO.title = title;
        newIMO.abstract = abstract;
        newIMO.ipfsHash = ipfsHash;
        newIMO.trainingSpec = TrainingSpec({
            modelArchitecture: modelArchitecture,
            modelCategory: modelCategory,
            datasetIds: datasetIds,
            maxSteps: maxSteps,
            mode: mode
        });
        newIMO.status = IMOStatus.Submitted;
        newIMO.quorumRequired = QUORUM_THRESHOLD;

        imoCount++;

        emit IMOProposed(newIMO.id, msg.sender, title, modelCategory);
    }

    /**
     * @dev Start voting phase — must be called before votes are cast.
     * Snapshots voter balances to prevent flash-loan vote manipulation.
     */
    function startVoting(uint256 imoId) external {
        IMO storage imo = imos[imoId];
        require(imo.status == IMOStatus.Submitted, "Not in submitted phase");
        imo.status = IMOStatus.Voting;
        imo.votingDeadline = block.timestamp + VOTING_PERIOD;
    }

    /**
     * @dev Register snapshot of token balance before voting.
     * Voters must call this during the voting period to lock their balance.
     * Balance is captured once and cannot be updated — prevents flash loans.
     */
    function snapshotBalance(uint256 imoId) external onlyRole(VOTER_ROLE) {
        IMO storage imo = imos[imoId];
        require(imo.status == IMOStatus.Voting, "Not in voting phase");
        require(block.timestamp <= imo.votingDeadline, "Voting ended");
        require(votingSnapshot[imoId][msg.sender] == 0, "Already snapshot");

        uint256 balance = IMO_TOKEN.balanceOf(msg.sender);
        require(balance >= MIN_STAKE, "Insufficient stake");
        votingSnapshot[imoId][msg.sender] = balance;
    }

    /**
     * @dev Cast a vote on an IMO using snapshotted balance.
     */
    function vote(uint256 imoId, bool support) external onlyRole(VOTER_ROLE) {
        IMO storage imo = imos[imoId];
        require(imo.status == IMOStatus.Voting, "Not in voting phase");
        require(block.timestamp <= imo.votingDeadline, "Voting ended");
        require(!imo.hasVoted[msg.sender], "Already voted");

        // Use snapshotted balance — prevents flash-loan vote manipulation
        uint256 stake = votingSnapshot[imoId][msg.sender];
        require(stake >= MIN_STAKE, "No snapshot or insufficient stake");

        imo.hasVoted[msg.sender] = true;
        imo.votes[msg.sender] = support;
        imoVoters[imoId].push(msg.sender);

        if (support) {
            imo.totalStake += stake;
        }

        emit VoteCast(imoId, msg.sender, support, stake);
    }

    /**
     * @dev Approve IMO if quorum reached
     */
    function resolveVoting(uint256 imoId) external {
        IMO storage imo = imos[imoId];
        require(imo.status == IMOStatus.Voting, "Not in voting phase");
        require(block.timestamp > imo.votingDeadline, "Voting not ended");

        if (imo.totalStake >= imo.quorumRequired) {
            imo.status = IMOStatus.Approved;
            emit IMOApproved(imoId);
        } else {
            imo.status = IMOStatus.Failed;
        }
    }

    /**
     * @dev Start training for an approved IMO
     */
    function startTraining(uint256 imoId) external {
        IMO storage imo = imos[imoId];
        require(imo.status == IMOStatus.Approved, "Not approved");

        imo.status = IMOStatus.Training;
        emit TrainingStarted(imoId);
    }

    /**
     * @dev Mark training as completed
     */
    function completeTraining(uint256 imoId) external {
        IMO storage imo = imos[imoId];
        require(imo.status == IMOStatus.Training, "Not in training");

        imo.status = IMOStatus.Completed;
        emit TrainingCompleted(imoId);
    }

    /**
     * @dev Get IMO details
     */
    function getIMO(uint256 imoId)
        external
        view
        returns (
            uint256 id,
            address proposer,
            string memory title,
            IMOStatus status,
            uint256 totalStake,
            uint256 votingDeadline,
            bool quorumReached
        )
    {
        IMO storage imo = imos[imoId];
        return (
            imo.id,
            imo.proposer,
            imo.title,
            imo.status,
            imo.totalStake,
            imo.votingDeadline,
            imo.totalStake >= imo.quorumRequired
        );
    }

    /**
     * @dev Get voting statistics
     */
    function getVotingStats(uint256 imoId)
        external
        view
        returns (uint256 yesVotes, uint256 noVotes, uint256 totalVoters)
    {
        IMO storage imo = imos[imoId];
        address[] storage voters = imoVoters[imoId];
        totalVoters = voters.length;

        for (uint256 i = 0; i < voters.length; i++) {
            if (imo.votes[voters[i]]) {
                yesVotes++;
            } else {
                noVotes++;
            }
        }
    }
}