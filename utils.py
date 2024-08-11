import numpy as np

def load_relevant_data_subset(df, ROWS_PER_FRAME):
    data_columns = ['x', 'y', 'z']
    data = df
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)

