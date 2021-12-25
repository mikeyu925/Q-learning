import numpy as np
import pandas as pd
if __name__ == "__main__":
    A = pd.DataFrame(
        data=np.array([1,2,2,4,4,3]).reshape(2,3),
        columns= ['+','*','-'],
    )
    print(A)
    A += 2
    print(A)