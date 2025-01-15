import numpy as np
from scipy.stats import expon, geom
from scipy.integrate import solve_ivp
from typing import Callable, List, Tuple, Union

from src.dynamical_sys import DynamicalSystem

class PredictabilityAnalyzer:
    def __init__(self, system: DynamicalSystem):
        """
        Initialize the predictability analyzer with a dynamical system.

        Parameters:
        ----------
        system : DynamicalSystem
            An instance of DynamicalSystem to analyze.
        """
        self.system = system

    def sample_trajectory(
        self, 
        Ttr: float, 
        T_sample: float, 
        n_samples: int
    ) -> np.ndarray:
        """
        Sample the trajectory of the system using a random sampling approach.

        Parameters:
        ----------
        Ttr : float
            Transient time (time to evolve before sampling).
        T_sample : float
            Total sampling time.
        n_samples : int
            Number of samples to take.

        Returns:
        -------
        np.ndarray
            Array of sampled states.
        """
        self.system.step(Ttr)
        samples = []

        if self.system.is_discrete:
            prob = n_samples / T_sample
            while len(samples) < n_samples:
                self.system.step(np.random.geometric(prob))
                samples.append(self.system.current_state().copy())
        else:
            beta = T_sample / n_samples
            while self.system.time < Ttr + T_sample:
                self.system.step(np.random.exponential(beta))
                samples.append(self.system.current_state().copy())

        return np.array(samples)

    def compute_predictability(
        self,
        Ttr: float = 200,
        T_sample: float = 1e4,
        n_samples: int = 500,
        d_tol: float = 1e-3,
        lambda_max: float = None,
        delta_range: np.ndarray = np.logspace(-9, -6, num=4),
        T_multiplier: float = 10,
        T_max: float = np.inf,
        nu_threshold: float = 0.5,
        C_threshold: float = 0.5,
    ) -> Tuple[str, float, float]:
        """
        Compute the predictability of the system using the method by Wernecke et al.

        Parameters:
        ----------
        Ttr : float, optional
            Transient time to evolve the system before sampling (default is 200).
        T_sample : float, optional
            Time to evolve the system for sampling (default is 1e4).
        n_samples : int, optional
            Number of samples for statistics calculation (default is 500).
        d_tol : float, optional
            Tolerance distance for Lyapunov prediction time calculation (default is 1e-3).
        lambda_max : float, optional
            Largest Lyapunov exponent (default is None; must be provided for chaotic systems).
        delta_range : np.ndarray, optional
            Array of perturbation distances to evaluate scaling (default is logspace(-9, -6)).
        T_multiplier : float, optional
            Multiplier for Lyapunov prediction time to get evaluation time (default is 10).
        T_max : float, optional
            Maximum evaluation time to consider (default is np.inf).
        nu_threshold : float, optional
            Threshold for cross-distance scaling coefficient (default is 0.5).
        C_threshold : float, optional
            Threshold for correlation coefficient (default is 0.5).

        Returns:
        -------
        Tuple[str, float, float]
            A tuple containing the chaos type (:SC, :PPC, :REG, :IND), the scaling coefficient nu, 
            and the correlation coefficient C.
        """
        samples = self.sample_trajectory(Ttr, T_sample, n_samples)
        mu = np.mean(samples, axis=0)
        s2 = np.mean([np.linalg.norm(s - mu) ** 2 for s in samples])

        distances = []
        correlations = []

        for delta in delta_range:
            Tlambda = np.log(d_tol / delta) / lambda_max if lambda_max and lambda_max > 0 else np.inf
            T = min(T_multiplier * Tlambda, T_max)
            
            d_sum, d2_sum = 0, 0
            for u in samples:
                n = np.random.normal(size=u.shape)
                n /= np.linalg.norm(n)
                u_hat = u + delta * n
                self.system.reinitialize(u)
                original = self.system.current_state().copy()
                self.system.reinitialize(u_hat)
                perturbed = self.system.current_state().copy()
                d = np.linalg.norm(original - perturbed)
                d_sum += d
                d2_sum += d**2
            
            d_avg = d_sum / len(samples)
            d2_avg = d2_sum / len(samples)
            C = 1 - d2_avg / (2 * s2)
            distances.append(d_avg)
            correlations.append(C)

        nu = np.polyfit(np.log(delta_range), np.log(distances), 1)[0]
        C_avg = np.mean(correlations)

        if nu > nu_threshold and C_avg > C_threshold:
            chaos_type = "REG"
        elif nu <= nu_threshold and C_avg > C_threshold:
            chaos_type = "PPC"
        elif nu <= nu_threshold and C_avg <= C_threshold:
            chaos_type = "SC"
        else:
            chaos_type = "IND"

        return chaos_type, nu, C_avg
