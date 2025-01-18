import numpy as np
import numba as nb

@nb.njit
def lorenz_system(state: np.ndarray, t: float, r: float, sigma: float = 10, beta: float = 8/3) -> np.ndarray:
    """
    Compute the derivatives of the Lorenz system at a given state.

    Parameters:
        state (List[float]): The current state [x, y, z] of the system.
        t (float): The current time (not used, but required by odeint).
        r (float): The parameter r in the Lorenz equations.
        sigma (float, optional): The parameter sigma in the Lorenz equations. Default is 10.
        beta (float, optional): The parameter beta in the Lorenz equations. Default is 8/3.

    Returns:
        np.array: The derivatives [dx/dt, dy/dt, dz/dt].
    """
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = (r - z) * x - y
    dzdt = x * y - beta * z
    return np.array([dxdt, dydt, dzdt], dtype=np.float64)

@nb.njit
def lorenz_system_ivp(t: float, state: np.ndarray, r: float, sigma: float = 10, beta: float = 8/3) -> np.ndarray:
    """
    Compute the derivatives of the Lorenz system at a given state.

    Parameters:
        state (List[float]): The current state [x, y, z] of the system.
        t (float): The current time (not used, but required by odeint).
        r (float): The parameter r in the Lorenz equations.
        sigma (float, optional): The parameter sigma in the Lorenz equations. Default is 10.
        beta (float, optional): The parameter beta in the Lorenz equations. Default is 8/3.

    Returns:
        np.array: The derivatives [dx/dt, dy/dt, dz/dt].
    """
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = (r - z) * x - y
    dzdt = x * y - beta * z
    return np.array([dxdt, dydt, dzdt], dtype=np.float64)