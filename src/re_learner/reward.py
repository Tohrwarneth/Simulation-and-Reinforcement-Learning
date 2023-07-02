class Reward:
    enterReward = 100
    leaveReward = 2 * enterReward
    notHomePenalty = -4 * enterReward + 4 * leaveReward
    illegalFloorPenalty = -30
