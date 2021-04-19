import numpy as np
def accuracy(y_hat, y):
    """
    Function to calculate the accuracy
    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the accuracy as float
    """
    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert(y_hat.size == y.size)
    matched=0
    for i in range(len(y_hat)):
        if(y_hat[i]==y[i]):
            matched+=1
    accuracy=(matched/len(y_hat))*100
    return accuracy

def precision(y_hat, y, cls):
    """
    Function to calculate the precision
    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the precision as float
    """
    True_positive=0
    False_positive=0
    assert(len(y_hat)==len(y))
    for i in range(len(y_hat)):
        if(y_hat[i]==cls):
            if(y[i]==cls):
                True_positive+=1
            else:
                False_positive+=1
    if(True_positive+False_positive==0):
        return 0
    else:
        precision=(True_positive/(True_positive+False_positive))*100
        return precision

def recall(y_hat, y, cls):
    """
    Function to calculate the recall
    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the recall as float
    """
    True_positive=0
    False_negative=0
    assert(len(y_hat)==len(y))
    for i in range(len(y_hat)):
        if(y[i]==cls):
            if(y_hat[i]==cls):
                True_positive+=1
            else:
                False_negative+=1
    if(True_positive+False_negative==0):
        return 0
    else:
        recall=(True_positive/(True_positive+False_negative))*100
        return recall

def rmse(y_hat, y):
    """
    Function to calculate the root-mean-squared-error(rmse)
    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the rmse as float
    """
    return np.sqrt(((y_hat - y) ** 2).mean())

def mae(y_hat, y):
    """
    Function to calculate the mean-absolute-error(mae)
    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the mae as float
    """
    return abs(y_hat-y).mean()