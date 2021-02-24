import pandas as pd
import numpy as np
column_name = ["a","b"]
row_name = ["a","b"]
matrix = np.zeros((2,2))

print(matrix)

df = pd.DataFrame(matrix,columns=column_name,index=row_name)
df.to_csv(r"./wins.csv")
print(df)