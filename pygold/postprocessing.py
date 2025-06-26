import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def postprocess_data(file_folder, targets=None):
    """
    Postprocesses data from a file directory containing results of a single
    or multiple algorithms. Generates the following plots:

    - ECDF plots showing the percentage of problems solved to
      each target accuracy per function evaluation budget
    - Performance profiles showing the performance of each solver
      relative to the best solver on each problem instance

    :param file_folder: Path or list of paths to the folder(s) each containing
        the data files for a single algorithm
    :param targets: Optional list of target accuracy values. Will default
        to [1e-1, 1e-2, 1e-4, 1e-8] if not provided.
    :return: Dictionary containing all plots.
    """
    # Read data from the file folder(s)
    df = read_coco_data(file_folder, targets)

    if df.empty:
        return {"error": "No valid data found in the specified folder(s)"}

    # Container for all results
    results = {
        "data": df,
        "plots": {}
    }

    # Generate ECDF plots
    results["plots"]["ecdf"] = plot_ecdf(df)

    # Number of unique solvers
    n_solvers = len(df['solver'].unique()) if 'solver' in df.columns else 0

    # Generate performance profiles if multiple solvers are present
    if n_solvers > 1:
        # Create a performance profile directly from the dataframe
        results["plots"]["performance_profiles"] = plot_performance_profiles(df)

    # Display the plots
    for plot_type, figs in results["plots"].items():
        if isinstance(figs, list):
            for fig in figs:
                fig.show()
        elif hasattr(figs, 'show'):
            figs.show()

    return results

def read_coco_data(file_folder, targets=None):
    """
    Reads data in the COCO format from a file directory.

    :param file_folder: Path or list of paths to the folder(s) containing the
        COCO data files for a single algorithm.
    :param targets: List of target accuracy values. If None, default targets
        will be used (1e-1, 1e-2, 1e-4, 1e-8).
    :return: A pandas DataFrame containing the data with columns
        ['solver', 'problem', 'n_dims', 'instance', 'target1', 'target2', ...]
        where the entries in the target correspond to the number of function
        evaluations to reach each target.
    """
    import os
    import re

    if targets is None:
        # Default targets if none provided (can be adjusted based on your needs)
        targets = [1e-1, 1e-2, 1e-4, 1e-8]

    records = []

    if isinstance(file_folder, str):
        file_folder = [file_folder]

    for folder in file_folder:
        # Find all .info files which contain metadata about the runs
        info_files = [f for f in os.listdir(folder) if f.endswith('.info')]

        for info_file in info_files:
            info_path = os.path.join(folder, info_file)

            # Parse the .info file to extract metadata
            with open(info_path, 'r') as f:
                info_content = f.readlines()

            # Extract solver, problem name, and dimensions from info file
            header = info_content[0]
            match = re.search(r'funcId\s*=\s*(\d+),\s*DIM\s*=\s*(\d+)', header)
            if match:
                func_id = int(match.group(1))
                n_dims = int(match.group(2))
            else:
                # Skip if we can't parse the function ID and dimension
                continue

            # Extract algorithm ID (solver name)
            match = re.search(r'algId\s*=\s*\'([^\']+)\'', header)
            solver = match.group(1) if match else os.path.basename(folder)

            # Extract problem name from the comment line (if available)
            problem_name = None
            if len(info_content) > 1 and info_content[1].startswith('%'):
                comment = info_content[1]
                # Extract problem name from comment like "% Run dual_annealing
                # on Bukin6 in 2D"
                match = re.search(r'on\s+(\w+)', comment)
                if match:
                    problem_name = match.group(1)

            if not problem_name:
                # If problem name wasn't found in comment, extract from filename
                name_match = re.search(r'(\w+)_f\d+_DIM', info_file)
                if name_match:
                    problem_name = name_match.group(1)
                else:
                    problem_name = f"f{func_id}"

            # Find data directory references in the .info file
            for line in info_content[2:]:
                if not line.strip() or line.startswith('%'):
                    continue

                # Parse data directory and instance information
                parts = line.split(',')
                if not parts:
                    continue

                data_file_ref = parts[0].strip()
                data_dir, data_file = os.path.split(data_file_ref)

                # Extract instances from the line
                # (e.g., "1:10272|63.7231697906152, 2:5445|63.7231697906152")
                instance_data = {}
                for instance_part in parts[1:]:
                    instance_match = re.search(r'(\d+):(\d+)\|([0-9.e+-]+)', instance_part)
                    if instance_match:
                        instance = int(instance_match.group(1))
                        fevals = int(instance_match.group(2))
                        best_fval = float(instance_match.group(3))
                        instance_data[instance] = {'fevals': fevals, 'best_fval': best_fval}

                # Find the corresponding .dat file to extract information
                dat_path = os.path.join(folder, data_dir, data_file.replace('.tdat', '.dat'))

                if os.path.exists(dat_path):
                    # Process the .dat file to extract function evaluation data
                    instance_convergence = {}
                    current_instance = None

                    with open(dat_path, 'r') as f:
                        for line in f:
                            line = line.strip()

                            # Skip empty lines
                            if not line:
                                continue

                            # Check for instance marker
                            if line.startswith('%% iter/random seed:'):
                                try:
                                    current_instance = int(line.split(':')[1].strip())
                                    instance_convergence[current_instance] = []
                                except (IndexError, ValueError):
                                    current_instance = None
                                continue

                            # Skip other comment lines
                            if line.startswith('%'):
                                continue

                            # Parse data line if we have a valid instance
                            if current_instance is not None:
                                try:
                                    parts = line.split()
                                    if len(parts) >= 3:
                                        fevals = int(parts[0])
                                        fval = float(parts[2])
                                        instance_convergence[current_instance].append((fevals, fval))
                                except (IndexError, ValueError):
                                    pass

                    # For each instance, find when target accuracies reached
                    for instance, convergence_data in instance_convergence.items():
                        if not convergence_data:
                            continue

                        # Create a record for this instance
                        record = {
                            'solver': solver,
                            'problem': problem_name,
                            'n_dims': n_dims,
                            'instance': instance,
                            'func_id': func_id
                        }

                        # Check when each target accuracy was reached
                        for i, target in enumerate(targets):
                            target_reached = False
                            for fevals, fval in convergence_data:
                                # Check if fval is within accuracy target
                                if abs(fval) <= target:
                                    record[f'target_{target}'] = fevals
                                    target_reached = True
                                    break

                            # If target wasn't reached, mark with NaN
                            if not target_reached:
                                record[f'target_{target}'] = np.nan

                        records.append(record)

    # Convert records to DataFrame
    df = pd.DataFrame(records)

    # Reorder columns for better presentation
    if not df.empty:
        base_cols = ['solver', 'problem', 'n_dims', 'instance', 'func_id']
        target_cols = [col for col in df.columns if col.startswith('target_')]
        df = df[base_cols + target_cols]

    return df

def plot_ecdf(df):
    """
    Plots the empirical cumulative distribution function (ECDF) of the
    percentage of problems solved to each target accuracy per function
    evaluation budget.

    :param df: A pandas DataFrame containing the data with columns
        ['solver', 'problem', 'n_dims', 'instance', 'target_x', 'target_y', ...]
        .
    :return: List of matplotlib figure objects containing the ECDF plots.
    """
    if df.empty:
        return []

    # Get target columns from dataframe if not specified
    target_cols = [col for col in df.columns if col.startswith('target_')]

    if not target_cols:
        return []

    # Group by solver
    grouped = df.groupby(['solver'])

    figures = []
    for solver, group in grouped:
        # Create a single plot with all targets for this solver
        fig, ax = plt.subplots(figsize=(10, 6))
        for target_col in sorted(target_cols, key=lambda x: float(x.replace('target_', ''))):
            target = float(target_col.replace('target_', ''))

            # Get evaluation counts for this target
            ecdf_data = group[target_col].values

            if len(ecdf_data) > 0:  # Only create plot if we have data
                ecdf_data = np.sort(ecdf_data)
                y_values = np.arange(1, len(ecdf_data) + 1) / len(ecdf_data)

                # Plot ECDF
                ax.step(ecdf_data, y_values, label=f'Target {target:.0e}', where='post')

        # Format solver name for display (replace underscores, capitalize words)
        solver_display = str(solver)
        if isinstance(solver, (tuple, list)):
            solver_display = solver[0]
        solver_display = solver_display.replace('_', ' ').title()
        ax.set_title(f'ECDF for {solver_display}')
        ax.set_xlabel('Function Evaluations')
        ax.set_ylabel('ECDF')
        ax.legend()
        ax.grid(True)

        # Use log scale for x-axis
        ax.set_xscale('log')

        plt.tight_layout()
        figures.append(fig)

    return figures

def plot_performance_profiles(df, tau_grid=None, metrics=None):
    """
    Plots the performance profiles of provided solvers.

    The performance profile for solver s is defined as:

    ρ_s(τ) = (1/|P|) * size{p ∈ P: r_{p,s} ≤ τ}

    where r_{p,s} is the performance ratio for solver s on problem p,
    and |P| is the total number of problems.

    :param df: A pandas DataFrame containing the solver performance data with
        columns ['solver', 'problem', 'n_dims', 'instance', 'target_x', ...].
    :param tau_grid: Optional list of values to evaluate the performance profile
        at. Defaults to points linearly spaced between 1 and 10.
    :param metrics: Optional string for the plot title.
    :return: The matplotlib figure object.
    """
    if df.empty:
        return None

    # Get unique solvers
    solvers = df['solver'].unique()
    if len(solvers) <= 1:
        return None

    # Get target columns
    target_cols = [col for col in df.columns if col.startswith('target_')]
    if not target_cols:
        return None

    # For each problem instance (problem, n_dims, instance),
    # find the first target reached by each solver
    problem_keys = df[['problem', 'n_dims', 'instance']].drop_duplicates()

    # Create performance data structure
    performance_data = {}

    for _, row in problem_keys.iterrows():
        problem = row['problem']
        n_dims = row['n_dims']
        instance = row['instance']
        key = (problem, n_dims, instance)
        performance_data[key] = {}

        # Get data for this problem instance
        prob_df = df[(df['problem'] == problem) &
                     (df['n_dims'] == n_dims) &
                     (df['instance'] == instance)]

        for _, solver_row in prob_df.iterrows():
            solver = solver_row['solver']

            # Find the first target reached (if any)
            metric = float('inf')
            success = False

            for target_col in target_cols:
                if not pd.isna(solver_row[target_col]):
                    metric = solver_row[target_col]
                    success = True
                    break

            performance_data[key][solver] = {'metric': metric, 'success': success}

    # Calculate performance ratios
    ratios = {}

    for key, solver_dict in performance_data.items():
        ratios[key] = {}

        # Find minimum metric among successful solvers
        min_metrics = [solver_dict[s]['metric'] for s in solver_dict
                      if solver_dict[s]['success']]

        if min_metrics:
            min_metric = min(min_metrics)

            # Calculate ratio for each solver
            for solver in solver_dict:
                if solver_dict[solver]['success']:
                    if min_metric == 0:
                        ratios[key][solver] = float('inf')
                    else:
                        ratios[key][solver] = solver_dict[solver]['metric'] / min_metric
                else:
                    ratios[key][solver] = float('inf')
        else:
            # If no solver succeeded, all ratios are infinite
            for solver in solver_dict:
                ratios[key][solver] = float('inf')

    # Generate tau grid if not provided
    if tau_grid is None:
        tau_grid = np.linspace(1, 10, 150)

    # Compute performance profiles
    profiles = {}
    n_problems = len(ratios)

    if n_problems == 0:
        return None

    for solver in solvers:
        # Get all ratios for this solver
        r_s = [ratios[key].get(solver, float('inf')) for key in ratios]

        # Calculate performance profile
        rho = [(np.sum(np.array(r_s) <= tau) / n_problems) for tau in tau_grid]
        profiles[solver] = (tau_grid, rho)

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    for solver, (tau, rho) in profiles.items():
        ax.plot(tau, rho, label=solver)
    ax.set_xlabel(r'$\tau$')
    ax.set_ylabel(r'Profile $\rho_s(\tau)$')
    title = 'Performance Profiles'
    if metrics is not None:
        title += f' ({metrics})'
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig
