import numpy as np
from scipy.stats import expon, geom

def sample_trajectory(trajectories, Ttr, T_sample, n_samples, system_type='continuous'):
    """
    Samples points from the given trajectories based on the Julia logic.
    
    Parameters:
        trajectories (ndarray): Full trajectory data with shape (n_points, state_dim).
        Ttr (float): Transient time to exclude at the start.
        T_sample (float): Total sampling time.
        n_samples (int): Approximate number of samples to take.
        system_type (str): 'continuous' or 'discrete' for the system type.
        
    Returns:
        sampled_trajectory (ndarray): Array of sampled points.
    """
    assert system_type in {'continuous', 'discrete'}, "Invalid system type. Choose 'continuous' or 'discrete'."
    
    # Remove transient points
    transient_points = int(Ttr / T_sample * len(trajectories))
    
    trajectories = trajectories[transient_points:]
    
    if system_type == 'continuous':
        # Exponential sampling for continuous systems
        beta = T_sample / n_samples
        inter_sample_times = expon(scale=beta).rvs(size=n_samples)
    else:
        # Geometric sampling for discrete systems
        p = n_samples / T_sample
        inter_sample_times = geom(p).rvs(size=n_samples)
    
    # Ensure sampling times are valid indices
    inter_sample_indices = np.cumsum(inter_sample_times).astype(int)
    inter_sample_indices = inter_sample_indices[inter_sample_indices < len(trajectories)]
    
    # Sample the trajectory
    sampled_trajectory = trajectories[inter_sample_indices]
    
    return sampled_trajectory