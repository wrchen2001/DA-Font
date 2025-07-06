import numpy as np

# Cyclize loader
def cyclize(loader):
    """Cyclize loader to create an infinite iterator over the data loader"""
    while True:
        for x in loader:
            yield x  # Use yield to return one sample at a time without restarting the loop


# Sample from [0, end) with (almost) equidistant interval
def uniform_indice(end, n_sample, duplicate=False, st=None):
    """Generate (approximately) uniformly distributed indices within [0, end)

    Args:
        end (int): Upper bound of the range (exclusive)
        n_sample (int): Number of samples to generate
        duplicate (bool): Allow duplicate indices if n_sample > end
        st (int or None): Starting offset, computed automatically if None

    Returns:
        np.ndarray: Array of sampled indices
    """
    if end <= 0:
        return np.empty(0, dtype=np.int)  # Return empty array if invalid range

    if not duplicate and n_sample > end:
        n_sample = end  # Prevent over-sampling when duplicates are not allowed

    # NOTE with endpoint=False, np.linspace does not sample the `end` value
    indice = np.linspace(0, end, num=n_sample, dtype=int, endpoint=False)

    if st is None and end:
        st = (end - 1 - indice[-1]) // 2  # Center the sampled indices

    return indice + st  # Apply starting offset


# Uniformly sample elements from a sequence
def uniform_sample(population, n_sample, st=None):
    """Uniformly sample elements from an ordered sequence (list, ndarray, or str)

    Args:
        population (list, np.ndarray, or str): Source sequence to sample from
        n_sample (int or None): Number of elements to sample
        st (int or None): Starting offset for sampling

    Returns:
        Same type as population: Sampled elements with uniform spacing
    """
    assert not isinstance(population, set), "population should have order"

    N = len(population)
    if n_sample is None:
        return population  # Return the full population if n_sample is not specified

    indice = uniform_indice(N, n_sample, st)

    if isinstance(population, np.ndarray):
        return population[indice]
    elif isinstance(population, list):
        return [population[idx] for idx in indice]
    elif isinstance(population, str):
        return ''.join([population[idx] for idx in indice])
    else:
        raise TypeError(type(population))  # Raise error for unsupported types