import numpy as np
import numba as nb

def generate_pairs_of_initial_states(n_pairs, distance):
    """Generate n pairs of initial states with the given distance
    
    Params.:
        n_pairs: int, number  of initial states to generate
        distance: float, distance

    Returns.:
        tuple: Two arrays of initial conditions. First array contains initial conditions for the first pair,
            and the second array contains initial conditions for the second pair.
    """

    initial_conditions_1 = np.random.rand(n_pairs, 3)

    # Generate the second set of initial conditions by adding the distance to the first set
    direction = np.random.normal(size=(n_pairs, 3))  # Random directions for the offset
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)  # Normalize directions
    initial_conditions_2 = initial_conditions_1 + direction * distance

    return initial_conditions_1, initial_conditions_2