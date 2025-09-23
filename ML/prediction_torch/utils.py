import numpy as np

def create_sequences_multi_step(X, y, context_len=45, horizon=7):
    X_seq, y_seq = [], []
    for i in range(len(X) - context_len - horizon + 1):
        X_seq.append(X[i:i+context_len])
        y_seq.append(y[i+context_len:i+context_len+horizon])
    return np.array(X_seq), np.array(y_seq)