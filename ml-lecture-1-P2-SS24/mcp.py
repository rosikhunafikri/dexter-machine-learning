import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import time
import random
import itertools
from numpy.random import default_rng
from tqdm import tqdm
from collections import OrderedDict

from helper_functions import get_radius, gen_lin_sep_dataset, inner_prod
from plot_functions import init_2D_linear_separation_plot, update_2D_linear_separation_plot

class MCPNeuron:
    def __init__(self, n_inputs, w=None, threshold=0):
        if w is None:
            self.w = np.zeros(n_inputs)
        else:
            assert len(self.w) == n_inputs, "Error, number of inputs must be the same as the number of weights."
            self.w = w
        self.threshold = threshold

    # P2.1 - Implement a forward-pass function for a simple MCP neuron
    def forward(self, x):
        """
        :param x: The input values
        :return: The output in the interval [-1,1]
        """
        # <START Your code here>

        y = np.sign(inner_prod(self.w,x) - self.threshold)

        if (y == 0):
            return 1


        # <END Your code here>
        return y # TODO: Overwrite this with your code

    # P.2.3 - Implement a function to randomize the weights and the threshold
    def set_random_params(self):
        # <START Your code here>
        self.threshold = np.random.uniform(-1,1,self.w)
        self.weight = np.random.uniform(-1,1)
        # TODO: remove
        # <END Your code here>


        


    # P.2.4 - Implement a function to check whether the MCP neuron represents a specific Boolean Function
    def is_bf(self, bf):
        fail = False
        for x in bf.keys():
            y = bf[x]
            pred = self.forward(x)
            if y != pred:
                fail = True
                break
        return not fail


# P2.2 - Implement a function that generates sets of random Boolean functions given a dimension.
def generate_boolean_functions(dim, max_samples=0):
    """
    :param dim: The input dimension, i.e., the number of Boolean inputs
    :param max_samples: The max. number of functions to return.
    This value is bounded by the possible number of functions for a given input dimension.
    For example, for dim=2, there can only be 16 Boolean functions.
    :return: The functions to return as dictionaries, where keys are tuples of inputs and values are outputs.
    """
    input_combos = list(itertools.product([-1,1], dim))

    input_functions_limit = 2 ** (2 ** dim)

    if max_samples <= 0 or max_samples > input_combos:
        max_samples = input_functions_limit

    bf = []

    for _ in range(max_samples): #iterate to find many different permutations of output

        function = {} #initialize dictionary called function

        for input in input_combos:
            output = np.random.choice(-1,1)
            function[input] = output

        bf.append(function)
    
    # <START Your Code Here>

    # <END Your Code Here>
    return bf

if __name__ == '__main__':
    # P.2.1 - Test the forward function
    mcp_1 = MCPNeuron(1)
    mcp_1.w = np.array([1])
    mcp_1.threshold = 1

    print("MCP_1: \n")
    print(mcp_1.forward([1]))



    mcp_2 = MCPNeuron(2)
    mcp_2.w = np.array([1, 1])
    mcp_2.threshold = 1
    print("MCP_2: \n")
    print(mcp_2.forward([-1, 1]))

    mcp_3 = MCPNeuron(3)
    mcp_3.w = np.array([1, 1, 1])
    mcp_3.threshold = 1
    print("MCP_3: \n")
    print(mcp_3.forward([-1, 1, -1]))

    # P.2.4 - Test your is_bf function
    bf_1 = ({(-1, -1): -1, (-1, 1): -1, (1,-1): -1, (1, 1): 1})
    bf_2 = ({(-1, -1): 1, (-1, 1): 1, (1, -1): -1, (1, 1): 1})
    mcp_1 = MCPNeuron(2)
    # TODO: For P.2.4: Replace the weights and the threshold provided here such that mcp_1 represents bf_1.
    #  That is, for all four inputs the output must be correct.
    mcp_1.w = np.array([1,1])
    mcp_1.threshold = 1
    for k in bf_1.keys():
        out = mcp_1.forward(k)
        if out == bf_1[k]:
            print("Correct!")
        else:
            print("Incorrect")

    mcp_2 = MCPNeuron(2)
    # TODO: For P.2.4: Replace the weights and the threshold provided here such that mcp_2 represents bf_2
    #  That is, for all four inputs the output must be correct.
    mcp_2.w = np.array([-1,1])
    mcp_1.threshold = 0
    for k in bf_2.keys():
        out = mcp_2.forward(k)
        if out == bf_2[k]:
            print("Correct!")
        else:
            print("Incorrect")


    # P2.5 - Implement an evaluation script to estimate how many Boolean functions can be approximated with a MCP neuron.
    n_inputs_to_test = range(1,5)

    # TODO: Overwrite this succ_rates list. This list should contain, for each number of inputs,
    #  the fraction of linear threshold functions, i.e., the number of Boolean functions that can be approximated
    #  with a MCP neuron. For comparison, the numbers provided in the lecture are given in the succ_rates_lecture list.
    succ_rates = [0] * len(n_inputs_to_test)
    succ_rates_lecture = [1, 0.88, 0.5, 0.06][:len(n_inputs_to_test)]
    # <START Your Code Here>

    one = 



    # <END Your Code Here>

    # Plot the approximation success rates from your experiments and from the lecture
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.plot(n_inputs_to_test, succ_rates)
    plt.plot(n_inputs_to_test, succ_rates_lecture)
    plt.savefig('approx.png')

    # Not part of the exercise but interesting: Plot how many guesses were required to find a MCP parameterization.
    # fig, ax = plt.subplots(figsize=(10, 10))
    # plt.plot(n_inputs_to_test, avg_guesses)
    # # plt.show()
    # plt.savefig('guesses.png')
    # print("Done")


