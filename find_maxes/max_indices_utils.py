def find_top_n_idx(arr, top_n = 10):
    maxes = []
    max_indices = sorted(range(len(arr)), key=lambda i: arr[i], reverse=True)[:top_n]
    for i in range(len(max_indices)):
        maxes.append(arr[max_indices[i]])
    return maxes, max_indices

