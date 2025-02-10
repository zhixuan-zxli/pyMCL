import numpy as np

def binsearchkw(data: np.ndarray, queries: np.ndarray) -> np.ndarray:
    # Suppose 'data' is your m-by-n array and it is lexicographically sorted.
    m, n = data.shape
    # Create a structured dtype with n fields
    dtype = np.dtype({"names": [f"f{i}" for i in range(n)],
                      "formats": n * [data.dtype] })
    # Convert the array into a structured array (view trick)
    structured_data = data.view(dtype).reshape(-1)

    # Suppose 'queries' is your q-by-n array of query rows.
    structured_queries = queries.view(dtype).reshape(-1)

    # Use searchsorted to find insertion indices.
    indices = np.searchsorted(structured_data, structured_queries)
    indices = np.minimum(indices, m-1)
    match = structured_data[indices] == structured_queries

    return np.where(match, indices, -1)
