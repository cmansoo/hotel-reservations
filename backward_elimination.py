import numpy as np
import pandas as pd
import statsmodels.api as sm

def backward_elimination(x: np.ndarray, y: np.ndarray, s: float) -> np.ndarray:
    """
    x: feature matrix
    y: target column
    s: significance level
    """
    # check if the first column is an array of ones with len == len(x)
    # if true then use the x that was input
    # if not true then append column of ones in front
    
    if np.all(x[:, 0] == 1):
        pass
    else:
        x = np.concatenate([np.ones((len(x), 1)), x], axis=1)
    
    
    n_variables = list(range(len(x[0])))
    
    # convert to dataframe and assign variable names
    y_df = pd.Series(data=y, name="target")
    cols = list(map(lambda n: "x"+ str(n), n_variables))
    x_df = pd.DataFrame(data=x, columns=cols)
    
    # to store indicies
    records = np.array(n_variables)
    
    # backward elimination
    for i in range(len(n_variables)):
        logit = sm.Logit(y_df, x_df).fit()
        pvalues = logit.pvalues
        max_p = max(pvalues)
        max_p_index = pvalues.argmax()
        if max_p > s:
            x_df = x_df.drop(columns=x_df.columns[max_p_index], axis=1)
            records = np.delete(records, max_p_index)
        else:
            break
            
    print(logit.summary())
    
    # drop intercept index; indicies - 1 to offset so it matches the variable shape
    records = np.delete(records, 0) - 1
    
    print("remaining variable indices", records)
    
    # convert to numpy and drop intercept column from x_df
    return (x_df.to_numpy()[:, 1:], records.tolist())
