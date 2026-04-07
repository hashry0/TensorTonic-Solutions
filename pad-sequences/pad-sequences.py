import numpy as np

def pad_sequences(seqs, pad_value=-1, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    if max_len == None:
        max_len = max(len(seq) for seq in seqs)
    reform_seq = []
    for seq in seqs:
        length = len(seq)
        f_seq = np.full(max_len, pad_value)
        if length > max_len:
            seq = seq[:max_len]
        f_seq[:length] = seq
        reform_seq.append(f_seq)
    reform_seq = np.array(reform_seq)
    return reform_seq


#[[1,2],[3,4,5],[6]]
seqs = [[1,2],[3,4,5],[6]]
print(pad_sequences(seqs = seqs, max_len = 3))