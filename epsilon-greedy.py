import random
import numpy as np
import matplotlib.pyplot as plt


def EpsilonGreedy(epsilon,values):
    
    """performs exploration/explitation
       input: int epsilon, array values
       output: int picked_arm
    """
    sample=np.random.binomial(n=1,p=epsilon)

    if sample==1:
        picked_arm=random.choice(list(enumerate(values)))[0]
    else:
        picked_arm=np.argmax(values)

    return picked_arm

def CalculateReward(true_prob):
    
    """draws a reward based on a
       bernoulli distribution with
       probability taken from environments
       i.e., true_prob
       input: float true_prob
       output: float reward
    """
    reward=np.random.binomial(n=1,p=true_prob)
    return reward

def UpdateReward(picked_arm):
    """calculates the average rewards
       and update the index picked_arm of
       the array values
       input: int picked_arm
       output: float values[picked_arm]
    """
    counts[picked_arm]+=1

    arm_count=counts[picked_arm]
    arm_value=values[picked_arm]
    true_prob=true_prob_values[picked_arm]
    
    reward=CalculateReward(true_prob)
    values[picked_arm]=arm_value*((arm_count- 1)/arm_count) + reward/arm_count
    
    return(values[picked_arm])


# Main Program #
    
n_arms=5
counts=np.zeros(n_arms,dtype=int)
values=np.zeros(n_arms)
true_prob_values=[.7,.4,.1,.9,.5]

for i in range(500):
    picked_arm=EpsilonGreedy(0.1,values)
    print(i,picked_arm)
    value_picked_arm=UpdateReward(picked_arm)
#    print(value_picked_arm)
#    print(values)
    ave_value=np.mean(values)
    plt.scatter(i,ave_value)
plt.show()
     
    



