import numpy as np

gamma = 0.99
def _discounted_return(rewards):
    """
        Helper function

        Input: list of rewards {r_0, r_1, ..., r_t', ... r_T} from a single rollout of length T

        Output: array where each index t contains sum_{t'=0}^T gamma^t' r_{t'}
    """

    rewards = np.asarray(rewards)
    discounts = gamma**np.arange(0,len(rewards))
    discounted_returns = list(np.full_like(None, np.sum(discounts * rewards), shape=rewards.shape))
    return discounted_returns

def _discounted_cumsum(rewards):
    """
        Helper function which
        -takes a list of rewards {r_0, r_1, ..., r_t', ... r_T},
        -and returns an array where the entry in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}
    """

    discounted_cumsums = []
    rewards = np.asarray(rewards)
    discounts = gamma**np.arange(0,len(rewards))
    for t in range(len(rewards)):
        discounted_cumsums.append(np.sum(discounts[:len(rewards)-t]*rewards[t:]))

    return discounted_cumsums

if __name__ == "__main__":
    rewards = [150, 200, 500, 1000, 239, 200, 1, 67584]
    print("DISCOUNTED RETURN")
    print(_discounted_return(rewards))
    print("DISCOUNTED CUMSUMS")
    print(_discounted_cumsum(rewards))