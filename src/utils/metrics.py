import numpy as np
import numba as nb
from tqdm import tqdm
from typing import Optional
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from concurrent.futures import ProcessPoolExecutor, as_completed


class Metrics:
    def __init__(self)->None:
        pass

    @staticmethod
    def cross_correlation_3d_jit(x1, x2):
        """
        Compute the cross-correlation function C12(t) for two 3D trajectories x1(t) and x2(t),
        optimized with Numba for performance.

        Parameters:
            x1 (ndarray): Trajectory 1 (NxM array, where N is the number of time steps).
            x2 (ndarray): Trajectory 2 (NxM array, where N is the number of time steps).

        Returns:
            float: Cross-correlation value C12(t) across all dimensions.
        """
        # Ensure the trajectories have the same shape
        if x1.shape != x2.shape:
            raise ValueError("Trajectories x1 and x2 must have the same shape")
        
        # Calculate means and standard deviations along each dimension
        mu1 = np.mean(x1, axis=0)  # Mean for x1 (3 values)
        mu2 = np.mean(x2, axis=0)  # Mean for x2 (3 values)
        s1 = np.std(x1, axis=0)    # Standard deviation for x1 (3 values)
        s2 = np.std(x2, axis=0)    # Standard deviation for x2 (3 values)
        
        # Compute deviations and cross-correlation for all dimensions
        delta_x1 = x1 - mu1
        delta_x2 = x2 - mu2
        cross_corr = np.mean(delta_x1 * delta_x2, axis=0) / (s1 * s2)
        
        # Return the average cross-correlation across all dimensions
        return np.mean(cross_corr)
    
    @staticmethod
    def poincare_section(
        trajectory: np.ndarray,
        control_param: float,
        dim: int = 2,
        mode: str = "min",
        condition_val: Optional[float] = None,
        condition_dim: Optional[int] = None,
        if_ode: bool = True
    ) -> None:
        """
        Extracts points on a Poincaré section based on the specified mode.

        Args:
            trajectory (np.ndarray): The system trajectory, an NxM array where N is the time steps 
                and M is the number of dimensions.
            control_param (float): The control parameter value to associate with the detected points.
            control_param_list (List[float]): A list to store the control parameter values 
                corresponding to detected points.
            vals_list (List[float]): A list to store the values of the specified dimension 
                at the detected points.
            dim (int, optional): The dimension to check for extrema or conditions. Defaults to 2.
            mode (str, optional): The mode of detection. Options are:
                - "max": Detect local maxima.
                - "min": Detect local minima.
                - "condition": Detect crossings of a specific value in a specific dimension.
            Defaults to "min".
            condition_val (Optional[float], optional): The value to check for crossings 
                (used only in "condition" mode). Defaults to None.
            condition_dim (Optional[int], optional): The dimension to check for crossings 
                (used only in "condition" mode). Defaults to None.
            if_ode (bool, optional): Whether the used solver is odeint or solve_ivp.

        Raises:
            AssertionError: If `dim` or `condition_dim` exceeds the number of dimensions in the trajectory.

        Returns:
            None: The results are stored in `control_param_list` and `vals_list`.
        """
        assert dim < trajectory.shape[1], "Dimension to check should be less than system dimension!"

        if condition_dim is not None:
            assert condition_dim < trajectory.shape[1], "Condition dimension should be less than system dimension!"

        if if_ode:
            vals = trajectory[:, dim]
        else:
            vals = trajectory[dim]
        control_param_list = []
        vals_list = []

        if mode == "max":
            # Detect local maxima
            maxima_indices = np.where((vals[1:-1] > vals[:-2]) & (vals[1:-1] > vals[2:]))[0] + 1
            control_param_list.extend([control_param] * len(maxima_indices))
            vals_list.extend(vals[maxima_indices])

        elif mode == "min":
            # Detect local minima
            minima_indices = np.where((vals[1:-1] < vals[:-2]) & (vals[1:-1] < vals[2:]))[0] + 1
            control_param_list.extend([control_param] * len(minima_indices))
            vals_list.extend(vals[minima_indices])

        elif mode == "condition" and condition_val is not None and condition_dim is not None:
            # Detect crossings of condition_val in condition_dim
            if if_ode:
                trajectory_dim = trajectory[:, condition_dim]
            else:
                trajectory_dim = trajectory[condition_dim]
            cross_indices = np.where(
                (trajectory_dim[:-1] <= condition_val) & (trajectory_dim[1:] >= condition_val) |
                (trajectory_dim[:-1] >= condition_val) & (trajectory_dim[1:] <= condition_val)
            )[0]
            control_param_list.extend([control_param] * len(cross_indices))
            vals_list.extend(vals[cross_indices])

        return control_param_list, vals_list
    
    @staticmethod
    def compute_distances_and_correlations(dynamics, 
                                       initial_conditions: np.ndarray, 
                                       delta_range: np.ndarray, 
                                       T: float, 
                                       d_tol: float, 
                                       lambda_max: float, 
                                       dt: float = 0.01,
                                       T_multiplier: float = 10):
        """
        Compute mean distances and cross-correlation for each delta in delta_range.
        
        Parameters:
            dynamics (callable): Function defining the system's dynamics `dx/dt = f(x, t)`.
            initial_conditions (ndarray): Array of sampled points from the system's trajectory.
            delta_range (ndarray): Array of perturbation distances.
            T (float): Maximum integration time for perturbations.
            d_tol (float): Distance tolerance used for scaling time.
            lambda_max (float): Largest Lyapunov exponent for the system.
            dt (float): Integration time step.
            T_multiplier (float): Multiplier for T to control the size of the perturbations.
        
        Returns:
            distances (list): Mean distances for each delta.
            correlations (list): Cross-correlations for each delta.
        """
        distances = []
        correlations = []

        # Calculate the variance of the sampled trajectory
        mu = np.mean(initial_conditions, axis=0)
        s2 = np.mean(np.linalg.norm(initial_conditions - mu, axis=1)**2)
        # s2 = np.mean([np.linalg.norm(x - mu)**2 for x in initial_conditions])
        # mu = np.mean(initial_conditions)
        # s2 = np.mean((initial_conditions - mu)**2)

        for delta in tqdm(delta_range, desc="Processing delta values"):
            T_lyap = np.log(d_tol / delta) / lambda_max  # Lyapunov prediction time
            evaluation_time = min(T, max(T_multiplier * T_lyap, 200))

            mean_distance = 0
            mean_square_distance = 0

            for u in initial_conditions:
                # Perturb the initial condition
                n = np.random.normal(size=u.shape)
                n /= np.linalg.norm(n)  # Normalize to unit vector
                perturbed_u = u + delta * n

                # Integrate both trajectories
                t_span = np.arange(0, evaluation_time, dt)
                reference_trajectory = odeint(dynamics, u, t_span)
                perturbed_trajectory = odeint(dynamics, perturbed_u, t_span)

                # Calculate the distance and accumulate
                final_reference = reference_trajectory[-1]
                final_perturbed = perturbed_trajectory[-1]
                distance = np.linalg.norm(final_reference - final_perturbed)
                mean_distance += distance
                mean_square_distance += distance**2

            # Calculate mean distance and correlation
            mean_distance /= len(initial_conditions)
            mean_square_distance /= len(initial_conditions)
            correlation = 1 - mean_square_distance / (2 * s2)

            distances.append(mean_distance)
            correlations.append(correlation)

        return distances, correlations
    @staticmethod
    def compute_distance(dynamics, u, delta, evaluation_time, dt, r):
        # Perturb the initial condition
        n = np.random.normal(size=u.shape)
        n /= np.linalg.norm(n)  # Normalize to unit vector
        perturbed_u = u + delta * n

        # Integrate both trajectories
        t_span = np.arange(0, evaluation_time, dt)
        reference_trajectory = odeint(dynamics, u, t_span, args=(r,))
        perturbed_trajectory = odeint(dynamics, perturbed_u, t_span, args=(r,))

        # Calculate the distance
        final_reference = reference_trajectory[-1]
        final_perturbed = perturbed_trajectory[-1]
        distance = np.linalg.norm(final_reference - final_perturbed)
        return distance, distance**2

    def compute_distances_and_correlations_paral(self, dynamics, 
                                                initial_conditions: np.ndarray, 
                                                delta_range: np.ndarray, 
                                                T: float, 
                                                d_tol: float, 
                                                lambda_max: float, 
                                                r: float, 
                                                dt: float = 0.01,
                                                T_multiplier: float = 10):
        """
        Compute mean distances and cross-correlation for each delta in delta_range.
        
        Parameters:
            dynamics (callable): Function defining the system's dynamics `dx/dt = f(x, t)`.
            initial_conditions (ndarray): Array of sampled points from the system's trajectory.
            delta_range (ndarray): Array of perturbation distances.
            T (float): Maximum integration time for perturbations.
            d_tol (float): Distance tolerance used for scaling time.
            lambda_max (float): Largest Lyapunov exponent for the system.
            r (float): Control parameter value.
            dt (float): Integration time step.
            T_multiplier (float): Multiplier for T to control the size of the perturbations.
        
        Returns:
            distances (list): Mean distances for each delta.
            correlations (list): Cross-correlations for each delta.
        """

        if lambda_max < 0:
            print("REG 1.0 1.0")  # Regular
            return

        distances = []
        correlations = []

        # Calculate the variance of the sampled trajectory
        mu = np.mean(initial_conditions, axis=0)
        s2 = np.mean(np.linalg.norm(initial_conditions - mu, axis=1)**2)
        # s2 = np.mean([np.linalg.norm(x - mu)**2 for x in initial_conditions])
        # mu = np.mean(initial_conditions)
        # s2 = np.mean((initial_conditions - mu)**2)

        for delta in tqdm(delta_range, desc="Processing delta values"):
            T_lyap = np.log(d_tol / delta) / lambda_max  # Lyapunov prediction time
            evaluation_time = min(T, max(T_multiplier * T_lyap, 200))
            mean_distance = 0
            mean_square_distance = 0

            with ProcessPoolExecutor() as executor:
                futures = [
                    executor.submit(self.compute_distance, dynamics, u, delta, evaluation_time, dt, r)
                    for u in initial_conditions
                ]
                for future in as_completed(futures):
                    try:
                        distance, square_distance = future.result()
                        mean_distance += distance
                        mean_square_distance += square_distance
                    except Exception as e:
                        print(f"Error computing distance: {e}")

            # Calculate mean distance and correlation
            mean_distance /= len(initial_conditions)
            mean_square_distance /= len(initial_conditions)
            correlation = 1 - mean_square_distance / (2 * s2)

            distances.append(mean_distance)
            correlations.append(correlation)

        return distances, correlations
    
    @staticmethod
    def log_log_model(delta, nu, const, shift):
        """
        Log-log model function for cross-distance correlation

        Parameters:
            delta (float): Perturbation distance.
            nu (float): Scaling exponent.
            const (float): Constant term.
            c (float): Shift term.

        Returns:
            float: Log-log model function value.
        """
        return const*delta**nu + shift

    
    def compute_distance_slope(self, delta_range, distances):
        """
        Compute the slope of distances in log-log space.

        Parameters:
            delta_range (ndarray): Array of perturbation distances.
            distances (list): Mean distances corresponding to each delta.

        Returns:
            slope (float): Scaling coefficient (nu).
        """
        popt, _ = curve_fit(self.log_log_model, delta_range, distances, p0=[0.001,5, 1], bounds=([0,0,0],[1, np.inf, np.inf]))
        slope, const, shift = popt

        return slope