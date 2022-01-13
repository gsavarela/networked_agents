'''Provides statistical test for distributions 

    * Compares joint policies for affinity
'''
import numpy as np

# for statiscal testing.
from scipy.special import rel_entr
from scipy.stats import kstest

def rel_entropy(centralized_joint_policy, decentralized_joint_policy):
    '''Computes the relative entropy for each state

        * Measures the affinity of the distributions:
            - X == Y --> 0
        * Summary: negative values are compensated by positive values.

        * relative entropy for observation x_i, y_i
            x_i log (x_i / y_i) if x_i >=0 and y_i > 0
            0                   if x_i = 0
            +Inf                otherwise
        * relative entropy for distributions X, Y:
            = rel_entropy(x_1, y_1) + ... + rel_entropy(x_n, y_n)
    '''
    print('')
    print("########## Relative Entropy ##########")
    n_states = len(centralized_joint_policy)
    central = to_density(centralized_joint_policy) 
    decentral = to_density(decentralized_joint_policy) 
    
    total = 0
    for i in range(n_states):
        test, theo = decentral[i, :],  central[i, :]
        entr = np.sum(rel_entr(test, theo))
        print(f'State-{i}: {entr}')
        total += entr
    print(f"Relative Entropy-All states: {total}##########")

def ks_test(centralized_joint_policy, decentralized_joint_policy):
    '''Computes the two sided Komogorov-Smirnoff distance test.

        H0: The policies of distributed and centralized are equal.
        H1: The policies of distributed and centralized are different.
    '''
    n_states = len(centralized_joint_policy)
    central = to_cdf(centralized_joint_policy) 
    decentral = to_cdf(decentralized_joint_policy)
    print('')
    print("########## Komogorov-Smirnoff distance test ##########")
    for i in range(n_states):
        test, theo = decentral[i, :], central[i, :]
        print(f'State-{i}: {kstest(test, theo, alternative="two-sided")}')
    
def to_cdf(joint_policy):
    '''Converts a joint-policy into CDFs'''
    return to_density(np.array(joint_policy)).cumsum(axis=1)

def to_density(joint_policy):
    '''Converts a joint-policy into a joint_policy'''
    jp = np.array(joint_policy) 
    return jp / jp.sum(axis=1, keepdims=True)
