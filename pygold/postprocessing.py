import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import re

def postprocess_data(file_folder, targets=None):
    """
    Postprocesses data from a file directory containing results of a single
    or multiple algorithms. Generates the following plots and saves them to
    ppfigures/:

    - ECDF plots showing the percentage of problems solved to
      each target accuracy per function evaluation budget
    - Performance profiles showing the performance of each solver
      relative to the best solver on each problem instance

    :param file_folder: Path or list of paths to the folder(s) each containing
        the data files for a single algorithm
    :param targets: Optional list of target accuracy values. Will default
        to [1e-1, 1e-2, 1e-4, 1e-8] if not provided.
    :return: Dictionary containing all plots and dataframe object
        (runtime table).
    """

    # Read data from folder(s)
    df = read_coco_data(file_folder, targets)
    if df.empty:
        return {"error": "No valid data found in the specified folder(s)"}

    results = {"data": df, "plots": {}}

    # Output directory
    outdir = "ppfigures"
    os.makedirs(outdir, exist_ok=True)

    # ECDF plots: one per solver
    solvers = df['solver'].unique() if 'solver' in df.columns else []
    ecdf_figs = []
    for solver in solvers:
        # Create figure for this plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot ECDF on axes
        plot_ecdf(df[df['solver'] == solver], ax=ax)

        # Save figure
        ecdf_figs.append(fig)
        fname = os.path.join(outdir, f"ecdf_{solver}.png")
        fig.savefig(fname)
    results["plots"]["ecdf"] = ecdf_figs

    # Performance profile (if multiple solvers)
    perf_fig = None
    if len(solvers) > 1:
        # Create figure for performance profiles
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot performance profiles on axes
        plot_performance_profiles(df, ax=ax)

        # Save figure
        perf_fig = fig
        fname = os.path.join(outdir, "performance_profiles.png")
        fig.savefig(fname)
        results["plots"]["performance_profiles"] = perf_fig

    # Return results dictionary containing data and plots
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

    if targets is None:
        # Default targets if none provided
        targets = [1e-1, 1e-2, 1e-4, 1e-8]

    records = []

    if isinstance(file_folder, str):
        file_folder = [file_folder]

    for folder in file_folder:
        # Find all .info files for metadata about the runs
        info_files = [f for f in os.listdir(folder) if f.endswith('.info')]

        for info_file in info_files:
            info_path = os.path.join(folder, info_file)

            # Parse the .info file to extract metadata
            with open(info_path, 'r') as f:
                info_content = f.readlines()

            # Extract funcid, and dimensions from info file
            header = info_content[0]
            match = re.search(r'funcId\s*=\s*(\d+),\s*DIM\s*=\s*(\d+)', header)
            if match:
                func_id = int(match.group(1))
                n_dims = int(match.group(2))
            else:
                continue

            # Extract algorithm ID
            match = re.search(r'algId\s*=\s*\'([^\']+)\'', header)
            solver = match.group(1) if match else os.path.basename(folder)

            # Extract problem name from the comment line
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

def plot_ecdf(df, ax=None):
    """
    Plot ECDF for all targets in the given DataFrame on the provided axes.

    :param df: A pandas DataFrame containing the data with columns
        ['solver', 'problem', 'n_dims', 'instance', 'target_x', 'target_y', ...]
        .
    :param ax: The matplotlib axes object to plot on.
    """
    if df.empty or ax is None:
        return

    # Get target columns from dataframe
    target_cols = [col for col in df.columns if col.startswith('target_')]

    if not target_cols:
        return

    # Plot ECDF for each target
    for target_col in sorted(target_cols, key=lambda x: float(x.replace('target_', ''))):
        target = float(target_col.replace('target_', ''))

        # Get evaluation counts for this target
        ecdf_data = df[target_col].dropna().values

        if len(ecdf_data) > 0:
            ecdf_data = np.sort(ecdf_data)
            y_values = np.arange(1, len(ecdf_data) + 1) / len(ecdf_data)

            # Plot ECDF
            ax.step(ecdf_data, y_values, label=f'Target {target:.0e}', where='post')

    # Format solver name for display
    solver = df['solver'].iloc[0] if 'solver' in df.columns and not df.empty else ''
    solver_display = str(solver).replace('_', ' ').title()
    ax.set_title(f'ECDF for {solver_display}')
    ax.set_xlabel('Function Evaluations')
    ax.set_ylabel('ECDF')
    ax.legend()
    ax.grid(True)

    # Use log scale for x-axis
    ax.set_xscale('log')

    plt.tight_layout()
    plt.close()

def plot_performance_profiles(df, ax=None, tau_grid=None):
    """
    Plot performance profiles for all solvers on the provided axes.

    The performance profile for solver s is defined as:

    ρ_s(τ) = (1/|P|) * size{p ∈ P: r_{p,s} ≤ τ}

    where r_{p,s} is the performance ratio for solver s on problem p,
    and |P| is the total number of problems.

    A solver is considered successful on a problem if it reaches the
    hardest target for that problem instance.

    :param df: A pandas DataFrame containing the solver performance data with
        columns ['solver', 'problem', 'n_dims', 'instance', 'target_x', ...].
    :param ax: The matplotlib axes object to plot on.
    :param tau_grid: Optional list of values to evaluate the performance profile
        at. Defaults to points linearly spaced between 1 and 10.
    """
    if df.empty or ax is None:
        return

    # Get unique solvers
    solvers = df['solver'].unique()
    if len(solvers) <= 1:
        return

    # Get target columns and select the hardest (smallest) target
    target_cols = [col for col in df.columns if col.startswith('target_')]
    if not target_cols:
        return
    # Sort targets numerically and pick the smallest (hardest)
    hardest_col = sorted(target_cols, key=lambda x: float(x.replace('target_', '')))[-1]

    # For each problem instance (problem, n_dims, instance),
    # use the hardest target for success
    problem_keys = df[['problem', 'n_dims', 'instance']].drop_duplicates()
    performance_data = {}
    for _, row in problem_keys.iterrows():
        problem = row['problem']
        n_dims = row['n_dims']
        instance = row['instance']
        key = (problem, n_dims, instance)
        performance_data[key] = {}
        prob_df = df[(df['problem'] == problem) & (df['n_dims'] == n_dims) & (df['instance'] == instance)]
        for _, solver_row in prob_df.iterrows():
            solver = solver_row['solver']
            # Only use the hardest target for success
            metric = solver_row[hardest_col] if not pd.isna(solver_row[hardest_col]) else float('inf')
            success = not pd.isna(solver_row[hardest_col])
            performance_data[key][solver] = {'metric': metric, 'success': success}
    ratios = {}
    for key, solver_dict in performance_data.items():
        ratios[key] = {}
        min_metrics = [solver_dict[s]['metric'] for s in solver_dict if solver_dict[s]['success']]
        if min_metrics:
            min_metric = min(min_metrics)
            for solver in solver_dict:
                if solver_dict[solver]['success']:
                    ratios[key][solver] = solver_dict[solver]['metric'] / min_metric if min_metric != 0 else float('inf')
                else:
                    ratios[key][solver] = float('inf')
        else:
            for solver in solver_dict:
                ratios[key][solver] = float('inf')
    if tau_grid is None:
        tau_grid = np.linspace(1, 10, 150)
    profiles = {}
    n_problems = len(ratios)
    if n_problems == 0:
        return
    for solver in solvers:
        r_s = [ratios[key].get(solver, float('inf')) for key in ratios]
        rho = [(np.sum(np.array(r_s) <= tau) / n_problems) for tau in tau_grid]
        profiles[solver] = (tau_grid, rho)
    for solver, (tau, rho) in profiles.items():
        ax.plot(tau, rho, label=solver)
    ax.set_xlabel(r'$\tau$')
    ax.set_ylabel(r'Profile $\rho(\tau)$')
    title = 'Performance Profile'
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.close()
