import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("Output/labeled/q_values/dax/original/walk0_train_labeled.csv")

# Giả sử cột Q-values là 'iteration0_0', ..., 'iteration99_2'
qvalue_cols = [col for col in df.columns if col.startswith("iteration")]
qvalues = df[qvalue_cols].values

print("Min Q:", qvalues.min())
print("Max Q:", qvalues.max())
print("Mean Q:", qvalues.mean())
print("Std Q:", qvalues.std())

sample_qvalues = df.iloc[0][qvalue_cols].values.reshape(100, 3)
for q in sample_qvalues:
    plt.plot(q)
plt.title("Q-values of all experts for one sample")
plt.show()