import network as net
from random import random,  uniform


def main():

    """

        P1 and P2 are the given examples A is the most cottect answert
        and C is the most incorrect answer.P3 is test case.

    """
    bias = 1.0
    
    P1 = [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, bias]

    P2 = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, bias]

    P3 = [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, bias]

    weights = [uniform(-1, 1) for x in range(36)]
    coefficient = 0.002  # coefficient of learing 
    marginE = 0.0001     # margin of error

    # intialization of object
    obj = net.Network(weights, coefficient)
 
    e = 0.0
    # process of learning. If our start variable e,  as an error will be smaller than margin - then our net is quazi-intelligent 
    while True:
        e = 0.0
        rnd = random() % 2
        
        # for each loop,  e is increased in learning process.
        if rnd == 0:
            e += obj.learn(P1, 1.0)
            e += obj.learn(P2, 0.0)     
        else:
            e += obj.learn(P2, 0.0)
            e += obj.learn(P1, 1.0)
        if e < marginE:
            break
             
    # output for learned network
    print(obj.calc(P1))
    print(obj.calc(P2))
    print(obj.calc(P3))

if __name__ == "__main__":
    main()
