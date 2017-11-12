import numpy as np

def sigmoid(i):
    """ Applies a sigmoid non-linearity function. 
    Input: a 1-D NumPy array or a float
    Output: a 1-D NumPy array or integer
    """
    return np.round(1 / (1 + np.exp(-i)))

def add_bias(x):
    """ Adds a bias term to input data.
    Input: a 2-D Numpy array
    Output: a 2-D Numpy array with bias added
    """
    num = x.shape[1]
    ones = np.ones(num)
    return np.vstack((ones, x))

def activation(x, theta):
    """ Adds the bias term, calculates the linear result then applies the sigmoid function.
    Input: A 2-D NumPy array (the X matrix) and a 1-D NumPy array (the weights)
    Output: A 1-D NumPy array (predictions)
    """
    x = add_bias(x)
    lin_result = np.sum(x.T * theta, axis=1) 
    return sigmoid(lin_result)

def or_node(x):
    """ Applies weights that meet the requirements of the logical OR condition.
    Input: a 2-D NumPy array (the X matrix)
    Output: a 1-D NumPy array (predictions)
    """
    theta = np.array([-10, 20, 20])
    return activation(x, theta)

def and_node(x):
    """ Applies weights that meet the requirements of the logical AND condition.
    Input: a 2-D NumPy array (the X matrix)
    Output: a 1-D NumPy array (predictions)
    """
    theta = np.array([-30, 20, 20])
    return activation(x, theta)

def not_node(x):
    """ Applies weights that meet the requirements of the logical NOT AND condition.
    Input: a 2-D NumPy array (the X matrix)
    Output: a 1-D NumPy array (predictions)
    """
    theta = np.array([10, -20, -20])
    return activation(x, theta)

def xnor_net(x):
    """ Applies a simple Neural Network (weights are pre-set) to meet the requirements of the logical X NOR condition.
    Input: a 2-D NumPy array (the X matrix)
    Output: a 1-D NumPy array (predictions)
    """
    node_1 = and_node(x)
    node_2 = not_node(x)
    l2_input = np.vstack((node_1, node_2))
    return or_node(l2_input)

if __name__ == '__main__':
    x = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
    y = np.array([1, 0, 0, 1])
    print "When building statistical models, the logical X Nor is a challenging decision boundary to create."
    print "Here, we have data that includes the following examples: (False, False), (False, True), (True, False), and (True, True)"
    print ""
    print "These are saved as a numpy array X:"
    print x
    print ""
    print "The logical X Nor would predict our first and last sample to be True, and the others to be False:"
    print y
    print "" 
    print "Applying Logistic Regression to this problem, we can get a solution for (x1 OR x2), (NOT x1 AND NOT x2), or (x1 AND x2)."
    print ""
    print "Logistic Regression cannot get a solution for (NOT x1 AND NOT x2) OR (x1 AND x2), because the decision boundary is no longer linear"
    print ""
    print "Neural Networks can apply multiple 'nodes', each of which can be thought of as a logistic regression function. By simply combining "
    print "the three logistic regression examples mentioned above, we can reach our desired solution:"
    print ""
    print "xnor_net(x): "
    print xnor_net(x)



