from random import uniform
import math
import sys


# output of e is counted by sigmoid function
def sigmoid(x):

    return (1)/(1 + math.exp(-x))


class Network:

    """

        This is the Network class for computing network inputs,and weights

    """

    def __init__(self, weights, coefficient):
        self.weights = weights
        self.coef = coefficient

    def learn(self, arr, answer):
        out = 0.0
        if(len(arr) != len(self.weights)):
            print("Size of inputs and their wages doesn't matches!")
            sys.exit()

        for x in range(len(arr)):
            out += arr[x] * self.weights[x]

        y = sigmoid(out)

        for x in range(len(self.weights)):
            self.weights[x] += self.coef * (answer-y)*(1.0-y)*arr[x]*y

        e = (0.5)*(answer-y)*(answer-y)
        return e  # result of learning process as the expected margin of error

    def calc(self, array):
        out = 0.0
        for x in range(len(array)):
            out += self.weights[x] * array[x]
        
        return sigmoid(out)
