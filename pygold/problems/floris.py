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
        self.pdf = gaussian_kde(np.vstack([wind_direction, wind_speed]))

        # Generate a grid of wind directions and speeds
        n_pts = 10
        grid_x, grid_y = np.mgrid[0:360:n_pts*1j, 0:25:n_pts*1j]
        grid = np.vstack([grid_x.ravel(), grid_y.ravel()])

        # Evaluate the PDF at the grid points, save each point and the pdf value
        pdf_values = self.pdf(grid)

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
        2 * turbine diameter apart (252m). This is a pairwise distance
        constraint for all turbines.

        Returns an array of constraint values. Constraint value is negative if
        the constraint is violated.

        :param x: Input points (array-like, flat format: [x1, y1, x2, y2, ...])
        :return: Scalar constraint output
        """
        x = np.array(x).reshape(self.n_turbines, -1)

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

        :param x: Input points (array-like, flat format: [x1, y1, x2, y2, ...])
        :return: Array of constraint values
        """
        x = np.array(x).reshape(self.n_turbines, -1)

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
    (1000,1000) is functionally the same as Turbine 2 at (0,0) and Turbine 1 at
    (1000,1000)).
    """
    # Dimensions per turbine
    DIM = 2

    def __init__(self, n_turbines=10):
        super().__init__(n_turbines)

    def evaluate(self, x):
        """
        Calculate the farm Annual Energy Production (AEP) with a given
        turbine layout x. x should be a flat vector of length n_turbines*2
        formatted as [x1, y1, x2, y2, ..., xN, yN].

        :param x: Input layout (array-like, flat format: [x1, y1, x2, y2, ...])
        :return: AEP in Wh of the Floris model at the given layout
        """
        # Reshape flat vector to (n_turbines, 2)
        x = np.array(x).reshape(self.n_turbines, 2)
        if x.shape != (self.n_turbines, 2):
            raise ValueError("Invalid shape for x.")

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
        Format: [(x1_min, x1_max), (y1_min, y1_max), (x2_min, x2_max),
        (y2_min, y2_max), ...]

        :return: List of (min, max) tuples for each variable
        """
        bounds_list = []
        for i in range(self.n_turbines):
            bounds_list.extend([(0, 1000), (0, 1000)])
        return bounds_list

class TurbineLayoutYaw(FlorisProblem):
    """
    Represents a wind farm layout optimization problem with (default) 10
    turbines and yaw control. The problem has 3*turbines variables: each
    turbine's (x, y) coordinates and fixed yaw angle. The yaw angle is defined
    as an offset angle from the current wind direction. Each turbine will
    have the same yaw angle for all wind directions.

    The goal is to maximize annual energy production (AEP) while enforcing a
    minimum distance constraint between turbines. A permutation constraint is
    included to avoid equivalent layouts caused by swapping turbine indices
    (e.g., Turbine 1 at (0,0) and Turbine 2 at (1000,1000) is functionally the
    same as Turbine 2 at (0,0) and Turbine 1 at (1000,1000)).
    """
    # Dimensions per turbine
    DIM = 3

    def __init__(self, n_turbines=10):
        super().__init__(n_turbines)

    def evaluate(self, x):
        """
        Calculate the farm Annual Energy Production (AEP) with a given
        turbine layout x. x should be a flat vector of length n_turbines*3
        formatted as [x1, y1, alpha1, x2, y2, alpha2, ..., xN, yN, alphaN],
        where (xi, yi) are the coordinates of turbine i and alpha_i is its yaw
        angle in degrees.

        :param x: Input layout (array-like, flat format: [x1, y1, alpha1, x2,
            y2, alpha2, ...])
        :return: AEP in Wh of the Floris model at the given layout
        """
        # Reshape flat vector to (n_turbines, 3)
        x = np.array(x).reshape(self.n_turbines, 3)
        if x.shape != (self.n_turbines, 3):
            raise ValueError(f"Invalid shape for x. Expected (n_turbines, 3), got {x.shape}")

        # Set the turbine positions in the Floris model
        self.model.set(layout_x=x[:, 0], layout_y=x[:, 1])

        # Get the number of wind directions in the model
        n_wind_directions = len(self.model.wind_directions)

        # Reshape yaw angles to be 2D: (n_turbines, n_wind_directions)
        # Each turbine will have the same yaw angle for all wind directions
        # Future expansion could include different yaw angles for each direction
        yaw_angles = np.tile(x[:, 2], n_wind_directions).reshape(n_wind_directions,-1)

        # Set the yaw angles
        self.model.set(yaw_angles=yaw_angles)

        # Run the model and get the AEP
        self.model.run()
        aep = self.model.get_farm_AEP(freq=self.freqs)

        return aep

    def bounds(self):
        """
        Returns the bounds of the problem.
        Each turbine can be placed anywhere in a 1000m x 1000m area,
        and yaw angles are between -45 and 45 degrees. Note that yaw angles
        are an offset from the wind direction.
        Format: [(x1_min, x1_max), (y1_min, y1_max), (alpha1_min, alpha1_max),
        ...]

        :return: List of (min, max) tuples for each variable
        """
        bounds_list = []
        for i in range(self.n_turbines):
            bounds_list.extend([(0, 1000), (0, 1000), (-45, 45)])
        return bounds_list

class TurbineLayoutStochastic(TurbineLayout):
    """
    A stochastic version of the TurbineLayout problem. The problem is made
    stochastic by adding noise to the sampling points of the
    wind speeds x directions pdf at each function evaluation rather than
    using a fixed set of wind speeds and directions. The new wind speeds and
    directions are assigned weights according to the original wind speed and
    direction PDF.

    There is currently no support for postprocessing solver data from stochastic
    problems.

    Represents a wind farm layout optimization problem with (default) 10
    turbines. The problem has 2*turbines variables: each turbine's (x, y)
    coordinates. The goal is to maximize annual energy production (AEP)
    while enforcing a minimum distance constraint between turbines.
    A permutation constraint is included to avoid equivalent layouts caused by
    swapping turbine indices (e.g., Turbine 1 at (0,0) and Turbine 2 at
    (1000,1000) is functionally the same as Turbine 2 at (0,0) and Turbine 1 at
    (1000,1000)).
    """

    def evaluate(self, x):
        """
        Calculate the farm Annual Energy Production (AEP) with a given
        turbine layout x. x should be a flat vector of length n_turbines*2
        formatted as [x1, y1, x2, y2, ..., xN, yN].

        :param x: Input layout (array-like, flat format: [x1, y1, x2, y2, ...])
        :return: AEP in Wh of the Floris model at the given layout
        """
        # Reshape flat vector to (n_turbines, 2)
        x = np.array(x).reshape(self.n_turbines, 2)
        if x.shape != (self.n_turbines, 2):
            raise ValueError(f"x must be of shape ({self.n_turbines}, 2), got {x.shape}")

        # Sample and set wind speeds and directions
        n_pts = 10
        grid_x, grid_y = np.mgrid[0:360:n_pts*1j, 0:25:n_pts*1j]

        # Add noise to grid points to make stochastic
        grid_x = np.clip(grid_x + np.random.normal(scale=1.8, size=grid_x.shape), 0, 360)
        grid_y = np.clip(grid_y + np.random.normal(scale=0.125, size=grid_y.shape), 0, 25)
        grid = np.vstack([grid_x.ravel(), grid_y.ravel()])

        # Evaluate the PDF at the grid points, save each point and the pdf value
        pdf_values = self.pdf(grid)

        # normalize the pdf values to sum to 1 so we can use them as frequencies
        freqs = pdf_values / np.sum(pdf_values)

        # Set the wind directions and speeds in the Floris model
        self.model.set(wind_directions=grid[0, :], wind_speeds=grid[1, :], turbulence_intensities=np.full_like(grid[1, :], 0.06))

        # Set the turbine positions in the Floris model
        self.model.set(layout_x=x[:, 0], layout_y=x[:, 1])

        # Run the model and get the AEP
        self.model.run()
        aep = self.model.get_farm_AEP(freq=freqs)

        return aep

class TurbineLayoutYawStochastic(TurbineLayoutYaw):
    """A stochastic version of the TurbineLayout problem. The problem is made
    stochastic by adding noise to the sampling points of the
    wind speeds x directions pdf at each function evaluation rather than using
    a fixed set of wind speeds and directions. The new wind speeds and
    directions are assigned weights according to the original wind speed and
    direction PDF.

    There is currently no support for postprocessing solver data from stochastic
    problems.

    Represents a wind farm layout optimization problem with (default) 10
    turbines and yaw control. The problem has 3*turbines variables: each
    turbine's (x, y) coordinates and yaw angle. The goal is to maximize annual
    energy production (AEP) while enforcing a minimum distance constraint
    between turbines. A permutation constraint is included to avoid equivalent
    layouts caused by swapping turbine indices (e.g., Turbine 1 at (0,0) and
    Turbine 2 at (1000,100) is considered the same as Turbine 2 at (0,0) and
    Turbine 1 at (1000,100)).
    """
    def evaluate(self, x):
        """
        Calculate the farm Annual Energy Production (AEP) with a given
        turbine layout x. x should be a flat vector of length n_turbines*3
        formatted as [x1, y1, alpha1, x2, y2, alpha2, ..., xN, yN, alphaN].

        :param x: Input layout (array-like, flat format: [x1, y1, alpha1, x2,
            y2, alpha2, ...])
        :return: AEP in Wh of the Floris model at the given layout
        """
        # Reshape flat vector to (n_turbines, 3)
        x = np.array(x).reshape(self.n_turbines, 3)
        if x.shape != (self.n_turbines, 3):
            raise ValueError(f"x must be of shape ({self.n_turbines}, 3), got {x.shape}")

        # Sample and set wind speeds and directions
        n_pts = 10
        grid_x, grid_y = np.mgrid[0:360:n_pts*1j, 0:25:n_pts*1j]

        # Add noise to grid points to make stochastic
        grid_x = np.clip(grid_x + np.random.normal(scale=1.8, size=grid_x.shape), 0, 360)
        grid_y = np.clip(grid_y + np.random.normal(scale=0.125, size=grid_y.shape), 0, 25)
        grid = np.vstack([grid_x.ravel(), grid_y.ravel()])

        # Evaluate the PDF at the grid points, save each point and the pdf value
        pdf_values = self.pdf(grid)

        # normalize the pdf values to sum to 1 so we can use them as frequencies
        freqs = pdf_values / np.sum(pdf_values)

        # Set the wind directions and speeds in the Floris model
        self.model.set(wind_directions=grid[0, :], wind_speeds=grid[1, :], turbulence_intensities=np.full_like(grid[1, :], 0.06))

        # Set the turbine positions in the Floris model
        self.model.set(layout_x=x[:, 0], layout_y=x[:, 1])

        # Get the number of wind directions in the model
        n_wind_directions = len(self.model.wind_directions)

        # Reshape yaw angles to be 2D: (n_turbines, n_wind_directions)
        # Each turbine will have the same yaw angle for all wind directions
        # Future expansion could include different yaw angles for each direction
        yaw_angles = np.tile(x[:, 2], n_wind_directions).reshape(n_wind_directions,-1)

        # Set the yaw angles
        self.model.set(yaw_angles=yaw_angles)

        # Run the model and get the AEP
        self.model.run()
        aep = self.model.get_farm_AEP(freq=freqs)

        return aep
