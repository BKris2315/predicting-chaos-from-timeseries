import numpy as np
import numba as nb
from typing import Optional

class Metrics:
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