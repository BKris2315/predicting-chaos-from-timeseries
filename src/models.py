import numpy as np
import numba as nb

class Models:
    @staticmethod
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

    @staticmethod
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
    
    @staticmethod
    def lorenz_vec(x, t, sigma=10, rho=180.68, beta=8/3):
        dx = sigma * (x[1] - x[0])
        dy = x[0] * (rho - x[2]) - x[1]
        dz = x[0] * x[1] - beta * x[2]
        return [dx, dy, dz]
    
    @staticmethod
    def lorenz_vec2(x, t, params = [10, 180.68, 8/3]):
        sigma, rho, beta = params
        return np.array([
            sigma * (x[1] - x[0]),
            x[0] * (rho - x[2]) - x[1],
            x[0] * x[1] - beta * x[2]
        ])

    def lorenz_jacobian(x, params = [10, 180.68, 8/3]):
        sigma, rho, beta = params
        return np.array([
            [-sigma, sigma, 0],
            [rho - x[2], -1, -x[0]],
            [x[1], x[0], -beta]
    ])