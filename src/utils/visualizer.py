import numpy as np
from typing import List
import matplotlib.pyplot as plt

class Visualizer:

    @staticmethod
    def plot_trajectory_3d(trajectory: np.ndarray, labels: List[str] = None) -> None:
        """
        Plots the 3D trajectory of a system.

        Args:
        -------
        trajectory : np.ndarray
            Array of shape (N, 3) representing the trajectory in 3D space.
        labels : List[str], optional
            Labels for the axes, by default None
        """
        fig = plt.figure(dpi=150)
        ax = fig.add_subplot(111, projection='3d')
        ax.xaxis.set_pane_color((1,1,1,1))
        ax.yaxis.set_pane_color((1,1,1,1))
        ax.zaxis.set_pane_color((1,1,1,1))
        
        if labels is not None:
            ax.set_xlabel(labels[0])
            ax.set_ylabel(labels[1])
            ax.set_zlabel(labels[2])

        else:
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')

        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])
        plt.show()

    @staticmethod
    def plot_trajectory_2D(traj_x: np.ndarray, 
                           traj_y: np.ndarray, 
                           labels: List[str] = None) -> None:
        """
        Plots the 2D trajectory of a system.

        Args:
        -------
        traj_x : np.ndarray
            Array representing the x-coordinates of the trajectory.
        traj_y : np.ndarray
            Array representing the y-coordinates of the trajectory.
        labels : List[str], optional
            Labels for the axes, by default None
        """
        plt.figure(figsize=(10,8), dpi = 150)
        if labels is not None:
            plt.xlabel(labels[0])
            plt.ylabel(labels[1])
        else:
            plt.xlabel('x')
            plt.ylabel('y')

        plt.plot(traj_x, traj_y, lw = 1.5)
        plt.show()