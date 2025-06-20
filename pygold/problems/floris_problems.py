import numpy as np
import pandas as pd
from floris import FlorisModel
from scipy.stats import gaussian_kde
import os

class FlorisProblem:
    """
    Base problem class for problems using the FLORIS package.
    The goal is to maximize annual energy production (AEP)
    while enforcing a minimum distance constraint between turbines.
    A permutation constraint is included to avoid equivalent layouts caused by
    swapping turbine indices (e.g., Turbine 1 at (0,0) and Turbine 2 at
    (1000,100) is considered the same as Turbine 2 at (0,0) and Turbine 1 at
    (1000,100)).
    """
    def __init__(self, n_turbines=10):
        # Initialize the FlorisModel
        package_dir = os.path.dirname(os.path.abspath(__file__))
        floris_yaml_path = os.path.join(package_dir, "data", "emgauss.yaml")
        self.model = FlorisModel(floris_yaml_path)

        # Set the number of turbines
        self.n_turbines = n_turbines

        # Calculate the wind PDF
        wind_data_path = os.path.join(package_dir, "data", "A2E_Hourly_Samples.csv")
        wind_df = pd.read_csv(wind_data_path).dropna()
        wind_speed = np.array(wind_df.wind_speed)
        wind_direction = np.array(wind_df.wind_direction)
        pdf = gaussian_kde(np.vstack([wind_direction, wind_speed]))

        # Generate a grid of wind directions and speeds
        n_pts = 10
        grid_x, grid_y = np.mgrid[0:360:n_pts*1j, 0:25:n_pts*1j]
        grid = np.vstack([grid_x.ravel(), grid_y.ravel()])

        # Evaluate the PDF at the grid points, save each point and the pdf value
        pdf_values = pdf(grid)

        # normalize the pdf values to sum to 1 so we can use them as frequencies
        self.freqs = pdf_values / np.sum(pdf_values)

        # Set the wind directions and speeds in the Floris model
        self.model.set(wind_directions=grid[0, :], wind_speeds=grid[1, :], turbulence_intensities=np.full_like(grid[1, :], 0.06))

        # Save max no wake power
        # This value can be used to normalize the AEP which can help some
        # solvers' convergence.
        self.model.run_no_wake()
        self.aep_no_wake = self.model.get_farm_AEP(freq=self.freqs) * self.n_turbines

    def dist_constraint(self, x):
        """
        Distance constraint to ensure that turbines are spaced at least
        2 * turbine diameter apart (252m).cThis is a pairwise distance
        constraint for all turbines.

        Returns an array of constraint values. Constraint value is negative if
        the constraint is violated.

        :param x: Input points (array-like, shape=(n_turbines, 2))
        :return: Scalar constraint output
        """
        if x.shape != (self.n_turbines, 2):
            raise ValueError(f"x must be of shape ({self.n_turbines}, 2), got {x.shape}")

        # Min dist is 2 * turbine diameter
        min_dist = 2 * 126.0

        n = x.shape[0]
        dists = np.sum((x[:, None, :] - x[None, :, :])**2, axis=-1)

        idx_i, idx_j = np.triu_indices(n, k=1)
        constraint_vals = (dists[idx_i, idx_j])/ min_dist**2 - 1

        return constraint_vals

    def perm_constraint(self, x):
        """
        Permutation constraint to reduce the search space and avoid equivalent
        layouts caused by swapping turbine indices.

        The constraint is satisfied if the x-coordinate of each turbine
        is greater than the x-coordinate of the previous turbine.

        Returns an array of constraint values. Constraint value is negative if
        the constraint is violated.

        :param x: Input points (array-like, shape=(n_turbines, 2))
        :return: Array of constraint values
        """

        if x.shape != (self.n_turbines, 2):
            raise ValueError(f"x must be of shape ({self.n_turbines}, 2), got {x.shape}")

        perm_constraint_vals = np.zeros(self.n_turbines - 1)

        for i in range(self.n_turbines - 1):
            perm_constraint_vals[i] = x[i + 1, 0] - x[i, 0]

        return perm_constraint_vals

    def constraints(self):
        """
        Returns the constraints of the problem.

        :return: List of constraint functions for this benchmark
        """
        return [self.dist_constraint, self.perm_constraint]

class TurbineLayout(FlorisProblem):
    """
    Represents a wind farm layout optimization problem with (default) 10
    turbines. The problem has 2*turbines variables: each turbine's (x, y)
    coordinates. The goal is to maximize annual energy production (AEP)
    while enforcing a minimum distance constraint between turbines.
    A permutation constraint is included to avoid equivalent layouts caused by
    swapping turbine indices (e.g., Turbine 1 at (0,0) and Turbine 2 at
    (1000,100) is considered the same as Turbine 2 at (0,0) and Turbine 1 at
    (1000,100)).
    """
    def __init__(self, n_turbines=10):
        super().__init__(n_turbines)

    def evaluate(self, x):
        """
        Calculate the farm Annual Energy Production (AEP) with a given
        turbine layout x.

        :param x: Input layout (array-like, shape=(n_turbines, 2))
        :return: AEP in Wh of the Floris model at the given layout
        """
        # Check that x is shape (n_turbines, 2)
        if x.shape != (self.n_turbines, 2):
            raise ValueError(f"x must be of shape ({self.n_turbines}, 2), got {x.shape}")

        # Set the turbine positions in the Floris model
        self.model.set(layout_x=x[:, 0], layout_y=x[:, 1])

        # Run the model and get the AEP
        self.model.run()
        aep = self.model.get_farm_AEP(freq=self.freqs)

        return aep

    def bounds(self):
        """
        Returns the bounds of the problem.
        Each turbine can be placed anywhere in a 1000m x 1000m area.

        :return: Array of shape (n_turbines, 2) with bounds for each turbine
        """
        return np.array([[[0, 1000] for i in range(2)] for j in range(self.n_turbines)])

class TurbineLayoutYaw(FlorisProblem):
    """
    Represents a wind farm layout optimization problem with (default) 10
    turbines and yaw control. The problem has 3*turbines variables: each
    turbine's (x, y) coordinates and yaw angle. The goal is to maximize annual
    energy production (AEP) while enforcing a minimum distance constraint
    between turbines. A permutation constraint is included to avoid equivalent
    layouts caused by swapping turbine indices (e.g., Turbine 1 at (0,0) and
    Turbine 2 at (1000,100) is considered the same as Turbine 2 at (0,0) and
    Turbine 1 at (1000,100)).
    """
    def __init__(self, n_turbines=10):
        super().__init__(n_turbines)

    def evaluate(self, x):
        """
        Calculate the farm Annual Energy Production (AEP) with a given
        turbine layout x.

        :param x: Input layout (array-like, shape=(n_turbines, 3))
        :return: AEP in Wh of the Floris model at the given layout
        """
        # Check that x is shape (n_turbines, 3)
        if x.shape != (self.n_turbines, 3):
            raise ValueError(f"x must be of shape ({self.n_turbines}, 3), got {x.shape}")

        # Set the turbine positions and yaw angles in the Floris model
        self.model.set(layout_x=x[:, 0], layout_y=x[:, 1], yaw_angles=x[:, 2])

        # Run the model and get the AEP
        self.model.run()
        aep = self.model.get_farm_AEP(freq=self.freqs)

        return aep

    def bounds(self):
        """
        Returns the bounds of the problem.
        Each turbine can be placed anywhere in a 1000m x 1000m area,
        and yaw angles are between 0 and 360 degrees.

        :return: Array of shape (n_turbines, 3) with bounds for each turbine
        """
        return np.array([[[0, 1000], [0, 1000], [0, 360]] for i in range(self.n_turbines)])
