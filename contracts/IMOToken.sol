// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Permit.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";

/**
 * @title IMOToken
 * @dev Platform native token for rewards and governance
 * 
 * Token Allocation (1 billion total supply):
 * - 40% Community Rewards Pool (quality-based distribution)
 * - 20% Treasury (protocol development)
 * - 15% Team (4-year vesting)
 * - 15% Investors (6-month lockup)
 * - 10% Ecosystem Fund
 * 
 * Key Design Principles:
 * - No upfront staking required for IMO submission
 * - Rewards based on final model quality, not participation
 * - Quality multipliers: poor (0x), fair (0.5x), good (1x), excellent (1.5x), breakthrough (2x)
 * - Community evaluates models through voting and benchmarks
 */
contract IMOToken is ERC20, ERC20Permit, AccessControl, ReentrancyGuard {
    bytes32 public constant MINTER_ROLE = keccak256("MINTER_ROLE");
    bytes32 public constant EVALUATOR_ROLE = keccak256("EVALUATOR_ROLE");
    bytes32 public constant TREASURY_ROLE = keccak256("TREASURY_ROLE");

    // Quality multipliers (scaled by 100 for precision)
    uint256 public constant QUALITY_BREAKTHROUGH = 200;  // 2.0x
    uint256 public constant QUALITY_EXCELLENT = 150;     // 1.5x
    uint256 public constant QUALITY_GOOD = 100;          // 1.0x
    uint256 public constant QUALITY_FAIR = 50;           // 0.5x
    uint256 public constant QUALITY_POOR = 0;            // 0x (no reward)
    
    // Minimum quality threshold for any rewards (50%)
    uint256 public constant MIN_QUALITY_THRESHOLD = 50;

    // Pool allocations
    uint256 public constant DATA_POOL_PERCENT = 40;   // 40%
    uint256 public constant COMPUTE_POOL_PERCENT = 50; // 50%
    uint256 public constant PAPER_POOL_PERCENT = 10;  // 10%

    // IMO session tracking
    struct IMOSession {
        address proposer;
        string modelCategory;
        string ipfsHash;
        uint256 basePool;           // Base reward pool
        uint256 qualityScore;       // 0-100
        uint256 qualityMultiplier;  // 0-200
        uint256 dataPool;
        uint256 computePool;
        uint256 paperPool;
        bool evaluated;
        bool distributed;
        uint256 evaluationTimestamp;
    }

    mapping(uint256 => IMOSession) public imoSessions;
    uint256 public imoCount;

    // Contribution tracking
    struct Contribution {
        address contributor;
        uint256 score;
        uint256 claimed;
        bool verified;
    }

    mapping(uint256 => mapping(address => Contribution)) public contributions;
    mapping(uint256 => address[]) public contributorList;
    mapping(uint256 => uint256) public totalContributionScore;

    // Benchmark results
    struct BenchmarkResult {
        string name;
        uint256 score;  // 0-100
    }

    struct Evaluation {
        uint256 imoId;
        BenchmarkResult[] benchmarks;
        uint256 communityRating;  // 0-100
        uint256 sotaComparison;   // 0-100 (100 = matches SOTA)
        uint256 codeQuality;      // 0-100
        uint256 documentationQuality; // 0-100
        address evaluator;
        uint256 timestamp;
    }

    mapping(uint256 => Evaluation) public evaluations;

    event IMOProposed(
        uint256 indexed imoId,
        address indexed proposer,
        string modelCategory
    );

    event IMOEvaluated(
        uint256 indexed imoId,
        uint256 qualityScore,
        uint256 qualityMultiplier
    );

    event RewardsDistributed(
        uint256 indexed imoId,
        address indexed contributor,
        uint256 amount
    );

    event ContributionRecorded(
        uint256 indexed imoId,
        address indexed contributor,
        uint256 score
    );

    constructor() ERC20("IMO Token", "IMO") ERC20Permit("IMO Token") {
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(MINTER_ROLE, msg.sender);
        _grantRole(EVALUATOR_ROLE, msg.sender);

        // Mint initial supply to treasury
        _mint(msg.sender, 1_000_000_000 * 10 ** decimals());
    }

    /**
     * @dev Submit new IMO proposal (NO STAKING REQUIRED)
     */
    function proposeIMO(
        string memory modelCategory,
        string memory ipfsHash
    ) external returns (uint256) {
        uint256 imoId = ++imoCount;
        
        imoSessions[imoId] = IMOSession({
            proposer: msg.sender,
            modelCategory: modelCategory,
            ipfsHash: ipfsHash,
            basePool: 0,  // Set when reward pool allocated
            qualityScore: 0,
            qualityMultiplier: 0,
            dataPool: 0,
            computePool: 0,
            paperPool: 0,
            evaluated: false,
            distributed: false,
            evaluationTimestamp: 0
        });

        emit IMOProposed(imoId, msg.sender, modelCategory);
        return imoId;
    }

    /**
     * @dev Allocate reward pool for an IMO (called by treasury)
     */
    function allocateRewardPool(
        uint256 imoId,
        uint256 basePool
    ) external onlyRole(TREASURY_ROLE) {
        require(imoId <= imoCount && imoId > 0, "Invalid IMO");
        require(!imoSessions[imoId].evaluated, "Already evaluated");

        imoSessions[imoId].basePool = basePool;

        // Transfer from treasury to contract
        _transfer(msg.sender, address(this), basePool);
    }

    /**
     * @dev Evaluate model quality and calculate rewards
     */
    function evaluateIMO(
        uint256 imoId,
        string[] memory benchmarkNames,
        uint256[] memory benchmarkScores,
        uint256 communityRating,
        uint256 sotaComparison,
        uint256 codeQuality,
        uint256 documentationQuality
    ) external onlyRole(EVALUATOR_ROLE) nonReentrant {
        require(imoId <= imoCount && imoId > 0, "Invalid IMO");
        require(!imoSessions[imoId].evaluated, "Already evaluated");
        require(benchmarkNames.length == benchmarkScores.length, "Array length mismatch");

        IMOSession storage session = imoSessions[imoId];
        
        // Calculate overall quality score (0-100)
        // Weights: 40% benchmarks, 25% community, 25% SOTA, 5% code, 5% docs
        uint256 benchmarkAvg = 0;
        if (benchmarkScores.length > 0) {
            for (uint256 i = 0; i < benchmarkScores.length; i++) {
                benchmarkAvg += benchmarkScores[i];
            }
            benchmarkAvg = benchmarkAvg / benchmarkScores.length;
        }

        uint256 overallScore = (
            (benchmarkAvg * 40) +
            (communityRating * 25) +
            (sotaComparison * 25) +
            (codeQuality * 5) +
            (documentationQuality * 5)
        ) / 100;

        // Determine quality multiplier
        uint256 multiplier;
        if (overallScore >= 95) {
            multiplier = QUALITY_BREAKTHROUGH;  // 2.0x
        } else if (overallScore >= 85) {
            multiplier = QUALITY_EXCELLENT;     // 1.5x
        } else if (overallScore >= 70) {
            multiplier = QUALITY_GOOD;          // 1.0x
        } else if (overallScore >= 50) {
            multiplier = QUALITY_FAIR;          // 0.5x
        } else {
            multiplier = QUALITY_POOR;          // 0x (no reward)
        }

        session.qualityScore = overallScore;
        session.qualityMultiplier = multiplier;
        session.evaluated = true;
        session.evaluationTimestamp = block.timestamp;

        // Calculate adjusted pools
        uint256 adjustedPool = (session.basePool * multiplier) / 100;
        session.dataPool = (adjustedPool * DATA_POOL_PERCENT) / 100;
        session.computePool = (adjustedPool * COMPUTE_POOL_PERCENT) / 100;
        session.paperPool = (adjustedPool * PAPER_POOL_PERCENT) / 100;

        // Store evaluation
        evaluations[imoId] = Evaluation({
            imoId: imoId,
            benchmarks: new BenchmarkResult[](benchmarkNames.length),
            communityRating: communityRating,
            sotaComparison: sotaComparison,
            codeQuality: codeQuality,
            documentationQuality: documentationQuality,
            evaluator: msg.sender,
            timestamp: block.timestamp
        });

        emit IMOEvaluated(imoId, overallScore, multiplier);
    }

    /**
     * @dev Record contributor's score
     */
    function recordContribution(
        uint256 imoId,
        address contributor,
        uint256 score,
        bool verified
    ) external onlyRole(EVALUATOR_ROLE) {
        require(imoId <= imoCount && imoId > 0, "Invalid IMO");

        // Track new contributors
        if (contributions[imoId][contributor].contributor == address(0)) {
            contributorList[imoId].push(contributor);
        } else {
            // Update: subtract old score from total
            totalContributionScore[imoId] -= contributions[imoId][contributor].score;
        }

        contributions[imoId][contributor] = Contribution({
            contributor: contributor,
            score: score,
            claimed: 0,
            verified: verified
        });

        if (verified) {
            totalContributionScore[imoId] += score;
        }

        emit ContributionRecorded(imoId, contributor, score);
    }

    /**
     * @dev Calculate reward for a contributor
     */
    function calculateReward(
        uint256 imoId,
        address contributor,
        uint8 poolType
    ) public view returns (uint256) {
        IMOSession storage session = imoSessions[imoId];
        Contribution storage contribution = contributions[imoId][contributor];

        require(session.evaluated, "IMO not evaluated");
        require(contribution.verified, "Contribution not verified");

        // Get pool amount based on type (0=data, 1=compute, 2=paper)
        uint256 poolAmount;
        if (poolType == 0) {
            poolAmount = session.dataPool;
        } else if (poolType == 1) {
            poolAmount = session.computePool;
        } else if (poolType == 2) {
            poolAmount = session.paperPool;
        } else {
            return 0;
        }

        uint256 totalScore = totalContributionScore[imoId];
        if (totalScore == 0) return 0;

        return (contribution.score * poolAmount) / totalScore;
    }

    /**
     * @dev Claim rewards for a contributor
     */
    function claimRewards(
        uint256 imoId,
        uint8 poolType
    ) external nonReentrant {
        IMOSession storage session = imoSessions[imoId];
        require(session.evaluated, "IMO not evaluated");
        require(session.distributed, "Rewards not distributed");

        uint256 reward = calculateReward(imoId, msg.sender, poolType);
        require(reward > 0, "No rewards to claim");

        Contribution storage contribution = contributions[imoId][msg.sender];
        require(contribution.claimed < reward, "Rewards already claimed");

        uint256 toClaim = reward - contribution.claimed;
        contribution.claimed = reward;

        _transfer(address(this), msg.sender, toClaim);

        emit RewardsDistributed(imoId, msg.sender, toClaim);
    }

    /**
     * @dev Get IMO session details
     */
    function getIMO(uint256 imoId)
        external
        view
        returns (
            address proposer,
            string memory modelCategory,
            uint256 basePool,
            uint256 qualityScore,
            uint256 qualityMultiplier,
            uint256 dataPool,
            uint256 computePool,
            uint256 paperPool,
            bool evaluated
        )
    {
        IMOSession storage session = imoSessions[imoId];
        return (
            session.proposer,
            session.modelCategory,
            session.basePool,
            session.qualityScore,
            session.qualityMultiplier,
            session.dataPool,
            session.computePool,
            session.paperPool,
            session.evaluated
        );
    }

    /**
     * @dev Get quality multiplier for a score
     */
    function getQualityMultiplier(uint256 score) public pure returns (uint256) {
        if (score >= 95) return QUALITY_BREAKTHROUGH;
        if (score >= 85) return QUALITY_EXCELLENT;
        if (score >= 70) return QUALITY_GOOD;
        if (score >= 50) return QUALITY_FAIR;
        return QUALITY_POOR;
    }
}