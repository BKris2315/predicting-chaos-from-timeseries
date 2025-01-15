import numpy as np
from scipy.integrate import solve_ivp
from typing import Callable


class DynamicalSystem:
    def __init__(self, state: np.ndarray, dynamics: Callable, is_discrete: bool = False):
        """
        Initialize the dynamical system.

        Parameters:
        ----------
        state : np.ndarray
            Initial state of the system.
        dynamics : Callable
            Function defining the system's dynamics. 
            For continuous systems: f(state, t) -> np.ndarray
            For discrete systems: f(state) -> np.ndarray
        is_discrete : bool, optional
            True if the system is discrete, False if continuous (default is False).
        """
        self.state = np.array(state, dtype=float)
        self.dynamics = dynamics
        self.is_discrete = is_discrete
        self.time = 0.0  # For continuous systems

    def step(self, dt: float = 1.0) -> None:
        """
        Evolve the system by one step or time interval.

        Parameters:
        ----------
        dt : float, optional
            Time step for continuous systems (default is 1.0).
        """
        if self.is_discrete:
            self.state = self.dynamics(self.state)
        else:
            sol = solve_ivp(self.dynamics, [self.time, self.time + dt], self.state, t_eval=[self.time + dt])
            self.state = sol.y[:, -1]
            self.time += dt

    def current_state(self) -> np.ndarray:
        """
        Get the current state of the system.

        Returns:
        -------
        np.ndarray
            Current state of the system.
        """
        return self.state

    def reinitialize(self, state: np.ndarray) -> None:
        """
        Reinitialize the system's state.

        Parameters:
        ----------
        state : np.ndarray
            New initial state for the system.
        """
        self.state = np.array(state, dtype=float)
        self.time = 0.0