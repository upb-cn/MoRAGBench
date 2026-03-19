import random
from classes.common import SamplingMethod


def sample_items(task_name: str, items: list, limit: int, sampling_method: SamplingMethod, seed: int):
    """
    Sample items from a list using a specified method.

    Parameters
    ----------
    items : list
        The input list to sample from.
    limit : int
        Number of items to sample.
        If -1 → return all items.
    sampling_method : SamplingMethod
        One of {"random", "first_n", "last_n"}.
    seed : int
        Random seed used only for 'random' sampling.
    
    Returns
    -------
    tuple[list, list[int]]
        (sampled_items, sampled_indices)
    """

    n = len(items)
    
    # Handle limit = -1 → return everything
    if limit == -1 or limit >= n:
        if limit > n:
            print(
                f"WARN: limit ({limit}) for task {task_name} is larger than its size ({n}). "
                "Returning full list."
            )
        indices = list(range(n))
        return items, indices

     # Perform sampling
    if sampling_method == SamplingMethod.RANDOM:
        random.seed(seed)
        indices = random.sample(range(n), limit)
        sampled_items = [items[i] for i in indices]
        return sampled_items, indices

    elif sampling_method == SamplingMethod.FIRST_N:
        indices = list(range(limit))
        return items[:limit], indices

    elif sampling_method == SamplingMethod.LAST_N:
        indices = list(range(n - limit, n))
        return items[-limit:], indices

    else:
        raise ValueError(f"Unknown sampling method: {sampling_method}")
