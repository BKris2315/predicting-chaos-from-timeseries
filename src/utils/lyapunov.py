import numpy as np
from tqdm import tqdm 
from scipy.linalg import qr
from scipy.integrate import odeint


class Lyapunov:
    def __init__(self) -> None:
        pass
    @staticmethod
    def log_slope(dynamics, 
                                  initial_state: np.ndarray, 
                                  t_span: np.ndarray, 
                                  initial_points: int =None,
                                  perturbation: float = 1e-8, 
                                  d_tol:float = 1e-3,
                                  std_tol = 0.4):
        """
        Compute the largest Lyapunov exponent by evolving perturbed states with the system dynamics.
        
        Parameters:
            dynamics (callable): Function defining the system's dynamics `dx/dt = f(x, t)`.
            initial_state (ndarray): Initial state of the system.
            t_span (ndarray): Array of time points for integration.
            initial_points (int): Number of initial points for slope estimation.
            perturbation (float): Initial perturbation distance (delta).
            d_tol (float): Tolerance distance for Lyapunov prediction time.
            std_tol (float): Tolerance standard deviation for Lyapunov prediction time.
        
        Returns:
            lambda_max (float): Largest Lyapunov exponent.
            T_lambda (float): Lyapunov prediction time.
        """
        if initial_points:
            assert initial_points < len(t_span), "The initial points for slope estimation has to be smaller than the time span"
        # Evolve the original trajectory
        trajectory = odeint(dynamics, initial_state, t_span)

        # Initialize a perturbed state
        perturbed_state = initial_state + perturbation * np.random.normal(size=len(initial_state))

        # Evolve the perturbed trajectory
        perturbed_trajectory = odeint(dynamics, perturbed_state, t_span)

        # Compute the distance between the trajectories at each time step
        distances = np.linalg.norm(trajectory - perturbed_trajectory, axis=1)
        
        # Avoid log of zero by filtering out invalid distances
        valid_indices = distances > 0
        log_distances = np.log(distances[valid_indices])
        time_points = t_span[valid_indices]
        
        if initial_points is None:
            start = 0
            stop = 1000
            step = 1000
            std_tol_tmp = np.inf

            while std_tol_tmp >= std_tol and stop <= len(log_distances):
                std_tol_tmp = np.std(log_distances[start:stop])
                start = stop
                stop += step
            initial_points = start
        # Fit a linear regression to the log of distances
        if len(log_distances) > 1:
            lambda_max = np.polyfit(time_points[:initial_points], log_distances[:initial_points], 1)[0]
        else:
            raise ValueError("Insufficient data points for Lyapunov exponent calculation.")

        # Compute the Lyapunov prediction time
        if lambda_max > 0:
            T_lambda = np.log(d_tol / perturbation) / lambda_max
        else:
            T_lambda = np.inf  # Regular system
        
        return lambda_max, T_lambda
    
    @staticmethod
    def lambda_rescale(reference_state, test_state, scaling_factor):
        """
        Rescale the distance between two states in a parallel dynamical system.
        
        Parameters:
            reference_state (ndarray): The reference trajectory state (u1).
            test_state (ndarray): The test trajectory state (u2).
            scaling_factor (float): The scaling factor to adjust the distance.
        
        Returns:
            ndarray: The rescaled test state.
        """
        return reference_state + (test_state - reference_state) / scaling_factor

    
    def benettin(self, dynamics, initial_state, T, dt=1, Ttr=0, d0=1e-9, 
                            d0_lower=None, d0_upper=None, inittest=None, show_progress=False):
        """
        Calculate the maximum Lyapunov exponent using the Benettin method with rescaling.
        
        Parameters:
            dynamics (callable): Function defining the system's dynamics `dx/dt = f(x, t)`.
            initial_state (ndarray): Initial state of the system.
            T (float): Total time for the calculation.
            dt (float): Time step for integration.
            Ttr (float): Transient time before measurement starts.
            d0 (float): Initial and rescaling distance between trajectories.
            d0_lower (float): Lower threshold for distance rescaling (default: 1e-3 * d0).
            d0_upper (float): Upper threshold for distance rescaling (default: 1e+3 * d0).
            inittest (callable): Custom initialization function for the test state.
                                Signature: `inittest(u1, d0) -> test_state`.
            show_progress (bool): Whether to show a progress bar.
        
        Returns:
            lyapunov_exponent (float): Largest Lyapunov exponent.
        """
        # Default thresholds for rescaling
        if d0_lower is None:
            d0_lower = 1e-3 * d0
        if d0_upper is None:
            d0_upper = 1e+3 * d0
        # assert Ttr < T "Transient time (Ttr) must be less than simulation time (T)"

        # Default initialization of test state
        if inittest is None:
            def inittest(u1, d0):
                return u1 + d0 / np.sqrt(len(u1)) * np.random.normal(size=len(u1))

        # Initialize reference and test trajectories
        reference_state = np.copy(initial_state)
        test_state = inittest(reference_state, d0)

        # Transient evolution
        if Ttr != 0:
            t_span = np.arange(0, Ttr, dt)
            reference_trajectory = odeint(dynamics, reference_state, t_span)
            test_trajectory = odeint(dynamics, test_state, t_span)
            reference_state = reference_trajectory[-1]
            test_state = test_trajectory[-1]

            distance = np.linalg.norm(reference_state - test_state)
            if distance < d0_lower or distance > d0_upper:
                test_state = self.lambda_rescale(reference_state, test_state, distance/d0)

        # Initialize variables for Lyapunov calculation
        total_time = 0
        lyapunov_sum = 0
        progress = tqdm(total=T, desc="Lyapunov Exponent Calculation", disable=not show_progress)
        distance = np.linalg.norm(reference_state - test_state)

        if distance < d0_lower or distance > d0_upper:
            raise ValueError(
                "After rescaling, the distance of reference and test states\n was not `d0_lower ≤ d ≤ d0_upper as expected.\n Perhaps you are using a dynamical system where the algorithm doesn't work."
                )
        if distance == 0:
            raise ValueError("The distance between reference and test states was 0 at the start.")
        # if 
            
        while total_time < T:
            while distance >= d0_lower and distance <= d0_upper:
                # Integrate for time step dt
                t_span = [0, dt]
                reference_trajectory = odeint(dynamics, reference_state, t_span)
                test_trajectory = odeint(dynamics, test_state, t_span)

                # Update states
                reference_state = reference_trajectory[-1]
                test_state = test_trajectory[-1]

                # Compute the distance between trajectories
                distance = np.linalg.norm(reference_state - test_state)
                if total_time > T:
                    break
                total_time += dt
                progress.update(dt)

            distance = np.linalg.norm(reference_state - test_state)
            test_state = self.lambda_rescale(reference_state, test_state, distance/d0)
            # Accumulate the logarithm of the rescaling factor
            lyapunov_sum += np.log(distance / d0)

            # Update total time
            total_time += dt
            progress.update(dt)
        
        distance = np.linalg.norm(reference_state - test_state)
        scaling_factor = distance / d0
        test_state = reference_state + (test_state - reference_state) / scaling_factor

        # Accumulate the logarithm of the rescaling factor
        lyapunov_sum += np.log(distance / d0)

        progress.close()

        # Compute the Lyapunov exponent
        lyapunov_exponent = lyapunov_sum / T
        return lyapunov_exponent
    
    def spectrum(self, dynamics, jacobian, initial_state, 
                     N: int = 100000, 
                     k: int = 3, 
                     dt: float = 0.001, 
                     Ttr: int = 10, 
                     show_progress: bool = False):
        """
        Calculate the spectrum of Lyapunov exponents for a dynamical system using tangent dynamics.
        
        Parameters:
            dynamics (callable): Function defining the system's dynamics `dx/dt = f(x, t)`.
            jacobian (callable): Function computing the Jacobian matrix `df/dx` at a given state `x`.
            initial_state (ndarray): Initial state of the system.
            N (int): Number of samples for the Lyapunov spectrum.
            k (int): Number of exponents to compute.
            dt (float): Determines the time step
            Ttr (int): Transient time before measurement starts

        Returns:
            lyapunov_spectrum (ndarray): Spectrum of Lyapunov exponents.
        """
        state_dim = len(initial_state)
        if k is None:
            k = state_dim
        elif k > state_dim:
            raise ValueError("Number of exponents (k) cannot exceed system dimension.")

        # Initialize reference state and deviation vectors
        reference_state = np.copy(initial_state)
        deviations = np.eye(state_dim)[:, :k]  # Orthonormal basis for k directions
        lyapunov_sums = np.zeros(k)

        # Transient evolution
        if Ttr > 0:
            t_span = np.linspace(0, Ttr, int(Ttr / dt))
            reference_trajectory = odeint(dynamics, reference_state, t_span)
            reference_state = reference_trajectory[-1]

            # Evolve deviations during transient
            for _ in range(int(Ttr / dt)):
                deviations = self.evolve_tangent_dynamics(jacobian, reference_state, deviations, dt)
                deviations, _ = qr(deviations)

        # Main QR decomposition loop
        progress = tqdm(total=N, desc="Lyapunov spectrum", disable=not show_progress)
        for _ in range(N):
            # Step reference trajectory
            t_span = [0, dt]
            reference_trajectory = odeint(dynamics, reference_state, t_span)
            reference_state = reference_trajectory[-1]

            # Evolve deviation vectors using tangent dynamics
            deviations = self.evolve_tangent_dynamics(jacobian, reference_state, deviations, dt)

            # Perform QR decomposition
            deviations, r = qr(deviations)

            # Accumulate logarithms of the diagonal elements of R
            lyapunov_sums += np.log(np.abs(np.diag(r)))

            progress.update(1)
        progress.close()

        # Compute the Lyapunov exponents
        total_time = N * dt
        return lyapunov_sums / total_time

    @staticmethod
    def evolve_tangent_dynamics(jacobian, reference_state, deviations, dt):
        """
        Evolve the deviation vectors using the tangent dynamics (Jacobian).
        """
        J = jacobian(reference_state)
        return deviations + dt * np.dot(J, deviations)