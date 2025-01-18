import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Union, Dict
from scipy.integrate import odeint, solve_ivp
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.models import lorenz_system, lorenz_system_ivp

# Move this outside to make it picklable
def solve_for_r(r, t_span, dt, initial_state, if_ode, method):
    t = np.arange(t_span[0], t_span[1], dt)
    if if_ode:
        sol = odeint(lorenz_system, initial_state, t, args=(r,))
    else:
        sol = solve_ivp(
            lorenz_system,
            t_span,
            np.array(initial_state),
            args=(r,),
            method=method,
            dense_output=True
        )
    return sol

def solve_lorenz(
    r_values: np.ndarray,
    t_span: Tuple[float, float],
    dt: float,
    initial_state: np.ndarray,
    if_ode: bool = True,
    method: str = 'RK45',
    init_update: bool = True
) -> Dict[str, Union[np.ndarray, List[float]]]:
    t = np.arange(t_span[0], t_span[1], dt)
    trajectories = []

    if init_update:
        # Sequential processing if initial states depend on previous results
        for r in tqdm(r_values):
            if if_ode:
                sol = odeint(lorenz_system, initial_state, t, args=(r,))
                initial_state = sol[-1]
            else:
                sol = solve_ivp(
                    lorenz_system,
                    t_span,
                    np.array(initial_state),
                    args=(r,),
                    method=method,
                    dense_output=True
                )
                initial_state = sol.y[:, -1]
            trajectories.append(sol)
    else:
        # Parallel processing
        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(solve_for_r, r, t_span, dt, initial_state, if_ode, method): r
                for r in r_values
            }
            for future in tqdm(as_completed(futures), total=len(r_values)):
                try:
                    trajectories.append(future.result())
                except Exception as e:
                    print(f"Error solving for r={futures[future]}: {e}")

    results = {
        'r_values': r_values,
        'trajectories': trajectories,
    }

    return results