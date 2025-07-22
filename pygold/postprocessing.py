import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import re
import warnings
# Ignore warnings about sharex/sharey when using multiple axes
warnings.filterwarnings("ignore", message="When passing multiple axes, sharex and sharey are ignored. These settings must be specified when creating axes.")

def postprocess_data(file_folder, targets=None, energy_file=None):
    """
    Postprocesses data from a file directory containing results of a single
    or multiple algorithms. Generates the following plots and saves them to
    ppfigures/:

    - ECDF plots showing the percentage of problems solved to
      each target accuracy per function evaluation budget
    - Performance profile showing the performance of each solver
      relative to the best solver. A solver is considered successful
      on a problem if it reaches the hardest target for that problem instance.
    - Bar chart showing the success rates of each solver per target accuracy.
    - Improvement plot showing the relative progress from a starting point. The
      generated plot shows the percentage of problems for which the solver
      achieved at least 90% up to 100% of the possible improvement.

    If energy_file is provided, also generates energy analysis plots:

    - Energy consumption by solver
    - Energy vs problem dimensions (if multiple dimensions present)
    - Energy usage per component by solver
    - Relative energy performance heatmap
    - System specifications

    Also generates a pandas DataFrame containing the mean number of
    function evaluations required to reach each target accuracy for each
    solver and problem among successful runs.

    If provided data is multi-dimensional, the results shown in the plots will
    be aggregated across all dimensions without normalization by dimension.

    :param file_folder: Path or list of paths to the folder(s) each containing
        the data files for a single algorithm
    :param targets: Optional list of target accuracy values. Will default
        to [1e-1, 1e-2, 1e-4, 1e-8] if not provided.
    :param energy_file: Optional path to energy data CSV file
    :return: Dictionary with keys 'plots' and 'data' containing all plots and
        mean fevals table.
    """

    # Read data from folder(s) and energy file
    df = read_data(file_folder, targets, energy_file)
    if df.empty:
        return {"error": "No valid data found in the specified folder(s)"}

    results = {"data": {}, "plots": {}}

    # Output directories
    outdir = "ppfigures"
    fevals_dir = os.path.join(outdir, "fevals")
    energy_dir = os.path.join(outdir, "energy")
    os.makedirs(fevals_dir, exist_ok=True)
    if energy_file is not None:
        os.makedirs(energy_dir, exist_ok=True)

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
        fname = os.path.join(fevals_dir, f"ecdf_{solver}.png")
        fig.savefig(fname, dpi=300, bbox_inches='tight')
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
        fname = os.path.join(fevals_dir, "performance_profiles.png")
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        results["plots"]["performance_profiles"] = perf_fig

    # Bar chart for success rates
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_success_rates(df, ax=ax)
    results["plots"]["success_rates"] = fig
    fname = os.path.join(fevals_dir, "success_rates.png")
    fig.savefig(fname, dpi=300, bbox_inches='tight')

    # Improvement plot
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_improvement(df, ax=ax)
    results["plots"]["improvement"] = fig
    fname = os.path.join(fevals_dir, "improvement.png")
    fig.savefig(fname, dpi=300, bbox_inches='tight')

    # Energy analysis plots
    if energy_file is not None and 'energy_consumed' in df.columns:
        # Grouped boxplots for all targets
        target_cols = [col for col in df.columns if col.startswith('target_')]
        n_targets = len(target_cols)
        if n_targets > 0:
            fig, ax = plt.subplots(1, n_targets, figsize=(6*n_targets, 6), sharey=True)
            plot_energy_by_solver(df, ax=ax)
            results["plots"]["energy_by_solver_all_targets"] = fig
            fname = os.path.join(energy_dir, "energy_by_solver_all_targets.png")
            fig.savefig(fname, dpi=300, bbox_inches='tight')

        # Energy component breakdown for hardest target
        fig, ax = plt.subplots(1, len(solvers), figsize=(6*len(solvers), 6))
        plot_energy_components(df, ax=ax)
        results["plots"]["energy_components_hardest"] = fig
        fname = os.path.join(energy_dir, "energy_components_hardest.png")
        fig.savefig(fname, dpi=300, bbox_inches='tight')

        # Energy vs dimensions (if multiple dimensions present)
        unique_dims = df['n_dims'].unique() if 'n_dims' in df.columns else []
        if len(unique_dims) > 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_energy_vs_dimensions(df, ax=ax)
            results["plots"]["energy_vs_dimensions"] = fig
            fname = os.path.join(energy_dir, "energy_vs_dimensions.png")
            fig.savefig(fname, dpi=300, bbox_inches='tight')

        # Relative energy performance (if multiple solvers)
        if len(solvers) > 1:
            fig, ax = plt.subplots(figsize=(14, 8))
            plot_relative_energy_heatmap(df, ax=ax)
            results["plots"]["relative_energy_heatmap"] = fig
            fname = os.path.join(energy_dir, "relative_energy_heatmap.png")
            fig.savefig(fname, dpi=300, bbox_inches='tight')

        # System specifications
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_system_specs(df, ax=ax)
        results["plots"]["system_specs"] = fig
        fname = os.path.join(energy_dir, "system_specs.png")
        fig.savefig(fname, dpi=300, bbox_inches='tight')

        # Store energy summary statistics
        summary_stats = df.groupby('solver').agg({
            'energy_consumed': ['mean', 'std', 'median', 'min', 'max'],
            'cpu_energy': 'mean',
            'gpu_energy': 'mean',
            'ram_energy': 'mean'
        })
        summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns]
        summary_stats.index = [str(s).replace('_', ' ').title() for s in summary_stats.index]
        results['data']['energy_summary'] = summary_stats

    # Table for mean fevals to reach each target among successful runs
    target_cols = [col for col in df.columns if col.startswith('target_')]
    df_mean_fevals = df.groupby(['problem', 'solver', 'n_dims'])[target_cols].mean().reset_index()
    results['data']['mean_fevals'] = df_mean_fevals

    # Return results dictionary containing data and plots
    return results

def read_data(file_folder, targets=None, energy_file=None):
    """
    Reads data stored in the COCO format from a file directory into a pandas
    DataFrame. Optionally includes CodeCarbon energy data if a filepath to
    a CodeCarbon energy CSV file is provided.

    :param file_folder: Path or list of paths to the folder(s) containing the
        COCO data files for a single algorithm.
    :param targets: List of target accuracy values. If None, default targets
        will be used (1e-1, 1e-2, 1e-4, 1e-8).
    :param energy_file: Optional path to energy data CSV file.
    :return: A pandas DataFrame containing the data with columns
        ['solver', 'problem', 'n_dims', 'instance', 'improvement',
        'total_evals', 'target1', 'target2', ...] and any available energy
        data columns. The entries in the improvement column correspond to
        one minus the percentage improvement of the solver on that problem
        iteration, calculated by (f(x0) - f(solver))/(f(x0) - f(min))
        where f(x0) is the initial function value, f(solver) is the best
        function value found by the solver, and f(min) is the minimum
        function value. The entries in the target correspond to the number
        of function evaluations to reach each target.
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
                match = re.search(r'\son\s+(\w+)', comment)
                if match:
                    problem_name = match.group(1)

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
                            'instance': instance - 1,  # Convert to 0-based index
                            'func_id': func_id
                        }

                        # For each instance, find the initial f(x0) - f(min)
                        # and best f(solver) - f(min) values
                        if convergence_data:
                            initial = convergence_data[0][1]
                            best = min(data[1] for data in convergence_data)
                            total_evals = max(data[0] for data in convergence_data)
                            record['total_evals'] = total_evals

                        # Calculate the improvement metric
                        if initial == best:
                            improvement = 0.0
                        else:
                            improvement = ((initial - best) / initial)
                        record['improvement'] = improvement

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
        base_cols = ['solver', 'problem', 'n_dims', 'instance', 'func_id', 'improvement', 'total_evals']
        target_cols = [col for col in df.columns if col.startswith('target_')]
        df = df[base_cols + target_cols]

    # Add energy data if provided
    if energy_file is not None and os.path.exists(energy_file):
        energy_df = pd.read_csv(energy_file)

        if not energy_df.empty:
            # Merge energy data with performance data
            if not df.empty:
                merge_cols = ['solver', 'problem', 'n_dims', 'instance']
                df = pd.merge(df, energy_df, on=merge_cols, how='outer')
            else:
                # If performance data is empty, just use energy data
                df = energy_df
    return df

def plot_ecdf(df, ax=None):
    """
    Generates ECDF plots for all targets in the given DataFrame on the provided
    axes. The ECDF plots show how quickly an algorithm reaches target accuracies
    across the provided test problems.

    :param df: A pandas DataFrame minimally containing columns
        ['solver', 'problem', 'n_dims', 'instance', 'target_x', 'target_y', ...]
        where target_x, target_y, etc. are the target accuracy columns.
    :param ax: The matplotlib axes object to plot on.
    """
    if df.empty or ax is None:
        warnings.warn("Dataframe is empty or axes object is None for plot_ecdf.")
        return

    required_cols = ['solver', 'problem', 'n_dims', 'instance']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        warnings.warn(f"Dataframe missing required columns for plot_ecdf: {missing_cols}")
        return

    df = df.copy()

    # Get target columns from dataframe
    target_cols = [col for col in df.columns if col.startswith('target_')]

    if not target_cols:
        warnings.warn("No target columns found for ECDF.")
        return

    # Consistent color palette
    colors = plt.get_cmap('Dark2').colors

    # Plot ECDF for each target
    for idx, target_col in enumerate(sorted(target_cols, key=lambda x: float(x.replace('target_', '')))):
        target = float(target_col.replace('target_', ''))
        ecdf_data = df[target_col].values
        if len(ecdf_data) > 0:
            ecdf_data = np.sort(ecdf_data)
            y_values = np.arange(1, len(ecdf_data) + 1) / len(ecdf_data)
            ax.step(ecdf_data, y_values, label=f'Target {target:.1e}', where='post',
                    color=colors[idx], linewidth=2.2)

    solver = df['solver'].iloc[0] if 'solver' in df.columns and not df.empty else ''
    solver_display = str(solver).replace('_', ' ').title()
    ax.set_title(f'ECDF for {solver_display}', fontsize=14)
    ax.set_xlabel('Function Evaluations', fontsize=12)
    ax.set_ylabel(r'% of Tests', fontsize=12)
    ax.legend(fontsize=10, title_fontsize=11, loc='best')
    ax.grid(True, linestyle='--', alpha=0.7)
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
        minimally columns ['solver', 'problem', 'n_dims', 'instance',
        'target_x', ...].
    :param ax: The matplotlib axes object to plot on.
    :param tau_grid: Optional list of values to evaluate the performance profile
        at. Defaults to points linearly spaced between 1 and 10.
    """
    if df.empty or ax is None:
        warnings.warn("Dataframe is empty or axes object is None for plot_performance_profiles.")
        return

    required_cols = ['solver', 'problem', 'n_dims', 'instance']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        warnings.warn(f"Dataframe missing required columns for plot_performance_profiles: {missing_cols}")
        return

    # Get unique solvers
    solvers = df['solver'].unique()
    if len(solvers) <= 1:
        warnings.warn("Not enough solvers for performance profiles. At least 2 required.")
        return

    df = df.copy()

    # Get target columns and select the hardest (smallest) target
    target_cols = [col for col in df.columns if col.startswith('target_')]
    if not target_cols:
        warnings.warn("No target columns found for performance profiles.")
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
    # Consistent color palette

    colors = plt.get_cmap('Dark2').colors

    for idx, solver in enumerate(solvers):
        r_s = [ratios[key].get(solver, float('inf')) for key in ratios]
        rho = [(np.sum(np.array(r_s) <= tau) / n_problems) for tau in tau_grid]
        profiles[solver] = (tau_grid, rho)
        ax.plot(tau_grid, rho, label=solver.replace('_', ' ').title(), color=colors[idx], linewidth=2.2)
    ax.set_xlabel(r'$\tau$', fontsize=12)
    ax.set_ylabel(r'Profile $\rho(\tau)$', fontsize=12)
    title = 'Performance Profile'
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10, title_fontsize=11, loc='best')
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.close()

def plot_success_rates(df, ax=None):
    """
    Plots the success rates of each solver per target on the provided axes.

    :param df: A pandas DataFrame minimally containing the
        columns ['solver', 'problem', 'n_dims', 'instance', 'target_x', ...].
    :param ax: The matplotlib axes object to plot on.
    """
    if df.empty or ax is None:
        warnings.warn("Dataframe is empty or axes object is None for plot_success_rates.")
        return

    required_cols = ['solver', 'problem', 'n_dims', 'instance']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        warnings.warn(f"Dataframe missing required columns for plot_success_rates: {missing_cols}")
        return

    # Get target columns from dataframe
    target_cols = [col for col in df.columns if col.startswith('target_')]
    if not target_cols:
        warnings.warn("No target columns found for success rates.")
        return

    df = df.copy()

    solvers = df['solver'].unique()

    # Compute success rates for each solver and target column
    success_rates = pd.DataFrame(index=solvers)
    for target_col in target_cols:
        # Success if not NaN
        rates = df.groupby('solver')[target_col].apply(lambda x: x.notna().mean())
        success_rates[target_col] = rates
    # Rename columns for display
    success_rates.columns = [f'Target {float(col.split("_")[1]):.1e}' if '_' in col else col.title() for col in success_rates.columns]

    # Rename solvers for display (replace _ with space, capitalize)
    success_rates.index = [str(s).replace('_', ' ').title() for s in success_rates.index]

    # Plot success rates
    colors = plt.get_cmap('Dark2').colors

    success_rates.plot(kind='bar', ax=ax, color=colors, edgecolor='black', width=0.8)
    ax.set_xlabel('Solver', fontsize=12)
    ax.set_ylabel('Success Rate', fontsize=12)
    ax.set_title('Success Rates by Solver and Target', fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.legend(title='Target', fontsize=10, title_fontsize=11, loc='best')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.setp(ax.get_xticklabels(), rotation=0, ha='center', fontsize=11)
    plt.tight_layout()
    plt.close()

def plot_improvement(df, ax=None, plot_points=None):
    """
    Plot the improvement of each solver on the provided axes.

    The improvement of each run is defined as the percentage of the possible
    improvement achieved by the solver, calculated as:
    improvement = (f(x0) - f(solver)) / (f(x0) - f(best))
    where f(x0) is the initial function value, f(solver) is the best
    function value found by the solver, and f(best) is the minimum function
    value. The plot shows improvement percentage on the x-axis and the
    percentage of problems solved to that improvement on the y-axis.

    :param df: A pandas DataFrame containing
        columns ['solver', 'problem', 'n_dims', 'instance', 'improvement'].
    :param ax: The matplotlib axes object to plot on.
    :param plot_points: Optional list of points to plot the improvement at.
        If None, defaults to [0.5, 0.75, 0.9, 0.99, 0.999, 0.999999, 1.0].
    """
    if df.empty or ax is None:
        warnings.warn("Dataframe is empty or axes object is None for plot_improvement.")
        return

    required_cols = ['solver', 'problem', 'n_dims', 'instance', 'improvement']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        warnings.warn(f"Dataframe missing required columns for plot_improvement: {missing_cols}")
        return

    df = df.copy()

    # Get unique solvers
    solvers = df['solver'].unique()

    if plot_points is None:
        tau_values = [0.5, 0.75, 0.9, 0.99, 0.999, 0.999999, 1.0]
    else:
        tau_values = plot_points

    # Consistent color palette
    colors = plt.get_cmap('Dark2').colors

    # Calculate percentages per solver and tau
    for idx, solver in enumerate(solvers):
        solver_data = df[df['solver'] == solver]
        improvement_counts = []
        for tau in tau_values:
            count = solver_data[solver_data['improvement'] >= tau].shape[0]
            improvement_counts.append(count)

        # Normalize by total number of problems
        total_problems = solver_data.shape[0]
        if total_problems > 0:
            improvement_counts = [count / total_problems for count in improvement_counts]
        else:
            improvement_counts = [0] * len(tau_values)

        ax.plot(range(len(tau_values)), improvement_counts, label=solver.replace('_', ' ').title(), marker='o', color=colors[idx])

    ax.set_xlabel('Percent of Possible Improvement Achieved', fontsize=12)
    ax.set_ylabel('Percentage of Problems', fontsize=12)
    ax.set_title('Improvement', fontsize=14)
    ax.legend(title='Solver', fontsize=10, title_fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.xticks(range(len(tau_values)), [f'{tau}' for tau in tau_values])
    plt.tight_layout()
    plt.close()

def plot_energy_by_solver(df, ax=None):
    """
    Plot grouped boxplots of energy usage to all available target accuracies for
    each solver. ax should be a matplotlib axes object of the size
    (1 row, n targets columns).

    These plots show the energy each algorithm requires to reach each
    target accuracy, among all runs that reached that target.

    All runs where the algorithm failed to reach the target are
    excluded from the plot.

    :param df: DataFrame containing energy data, including columns
        ['solver', 'energy_consumed', 'total_evals', 'os', 'cpu_model',
        'python_version', 'target_x', ...] where target_x, ... are the target
        accuracy columns.
    :param ax: Matplotlib axes object to plot on.
    """
    if df.empty or ax is None:
        warnings.warn("Dataframe is empty or axes object is None for plot_energy_by_solver.")
        return

    # Check if required columns exist
    required_cols = ['solver', 'energy_consumed', 'total_evals', 'os', 'cpu_model', 'python_version']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        warnings.warn(f"Dataframe missing required columns for plot_energy_by_solver: {missing_cols}")
        return

    target_cols = [col for col in df.columns if col.startswith('target_')]
    if not target_cols:
        warnings.warn("No target columns found.")
        return

    df = df.copy()

    # Apply display formatting to 'solver' column for plotting
    df['solver'] = df['solver'].apply(lambda s: str(s).replace('_', ' ').title())
    colors = plt.get_cmap('Dark2').colors

    plot_cols = []

    # Calculate energy usage per solver for each target
    for target_col in target_cols:
        col_name = f'energy_to_{target_col}'
        df[col_name] = np.where(df[target_col].notna() & df['total_evals'].notna() & (df['total_evals'] > 0), df['energy_consumed'] * (df[target_col] / df['total_evals']), np.nan)
        plot_cols.append(col_name)

    # Rename solver column for plotting
    df = df.rename(columns={'solver': 'Algorithm'})

    bp = df.boxplot(ax=ax, column=plot_cols, by='Algorithm', patch_artist=True, showfliers=False, medianprops=dict(color='black'), whiskerprops=dict(color='black'))

    for i, ax in enumerate(np.atleast_1d(bp)):
        title = ax.get_title()
        parts = title.split('_')
        # Format the target value for the title
        try:
            target_val = float(parts[-1])
            target_str = f"{target_val:.1e}"
        except Exception:
            target_str = ''.join(parts[1:])
        ax.set_title(f'Energy to Accuracy {target_str}')
        if i == 0:
            ax.set_ylabel('Energy Consumed (kWh)')
        for box_idx, patch in enumerate(ax.patches):
            color = colors[box_idx % len(colors)]
            patch.set_facecolor(color)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.5)
    system_info = df.iloc[0]
    system_title = f"{system_info['os']} | {system_info['cpu_model']} | Python {system_info['python_version']}"
    plt.suptitle(f'Energy Consumption by Target and Algorithm\n{system_title}', fontsize=16)
    plt.tight_layout()
    plt.close()

def plot_energy_components(df, ax=None):
    """
    Plot energy usage by component for the hardest target).
    Only includes runs where the solver reached the hardest target.
    :param df: DataFrame containing energy data minimally with columns
        ['solver', 'cpu_energy', 'ram_energy', 'gpu_energy', 'gpu_count',
        'total_evals', 'energy_consumed', 'os', 'cpu_model', 'python_version'].
    :param ax: matplotlib axes object to plot on. ax must be a 1D array of
        size (1, number of solvers).
    """
    if df.empty or ax is None:
        warnings.warn("Dataframe is empty or axes object is None for plot_energy_components.")
        return

    required_cols = ['solver', 'cpu_energy', 'ram_energy', 'gpu_energy', 'gpu_count', 'total_evals', 'os', 'cpu_model', 'python_version']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        warnings.warn(f"Dataframe missing required columns for plot_energy_components: {missing_cols}")
        return

    target_cols = [col for col in df.columns if col.startswith('target_')]
    if not target_cols:
        warnings.warn("No target columns found.")
        return

    df = df.copy()

    # Hardest target is the smallest value
    hardest_col = sorted(target_cols, key=lambda x: float(x.replace('target_', '')))[0]

    # CPU and RAM always present, GPU only if there is at least one GPU
    energy_components = ['cpu_energy', 'ram_energy']
    if df['gpu_count'].sum() != 0:
        energy_components.append('gpu_energy')

    solvers = df['solver'].unique()
    colors = plt.get_cmap('Dark2').colors

    for i, solver in enumerate(solvers):
        solver_data = df[(df['solver'] == solver) & (df[hardest_col].notna())]
        component_data = [solver_data[component] * solver_data[hardest_col] / solver_data['total_evals'] for component in energy_components]
        component_labels = [comp.replace('_energy', '').upper() for comp in energy_components]

        box_plot = ax[i].boxplot(component_data, labels=component_labels, patch_artist=True, showfliers=False, medianprops=dict(color='black'), whiskerprops=dict(color='black'))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        formatted_solver = str(solver).replace('_', ' ').title()
        ax[i].set_title(f'{formatted_solver}', fontsize=14)
        ax[i].set_ylabel('Energy Usage (kWh)', fontsize=12)
        ax[i].grid(True, alpha=0.3, axis='y')

    # Get system info for suptitle
    system_info = df.iloc[0]
    system_title = f"{system_info['os']} | {system_info['cpu_model']} | Python {system_info['python_version']}"

    plt.suptitle(f'Energy Component Breakdown by Solver to Accuracy {hardest_col.replace("target_", "")}\n{system_title}', fontsize=16)
    plt.tight_layout()
    plt.close()

def plot_energy_vs_dimensions(df, ax=None):
    """
    Plot total energy consumption vs problem dimensions for each solver.

    :param df: DataFrame containing energy data minimally with columns
        ['solver', 'energy_consumed', 'n_dims', 'os', 'cpu_model',
        'python_version'].
    :param ax: The matplotlib axes object to plot on
    """
    if df.empty or ax is None:
        warnings.warn("Dataframe is empty or axes object is None for plot_energy_vs_dimensions.")
        return

    # Check if required columns exist
    required_cols = ['solver', 'energy_consumed', 'n_dims', 'os', 'cpu_model', 'python_version']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        warnings.warn(f"Dataframe missing required columns for plot_energy_vs_dimensions: {missing_cols}")
        return

    df = df.copy()

    colors = plt.get_cmap('Dark2').colors

    for i, solver in enumerate(df['solver'].unique()):
        solver_data = df[df['solver'] == solver]
        avg_energy = solver_data.groupby('n_dims')['energy_consumed'].mean()
        std_energy = solver_data.groupby('n_dims')['energy_consumed'].std()

        formatted_solver = str(solver).replace('_', ' ').title()
        ax.errorbar(avg_energy.index, avg_energy.values, yerr=std_energy.values, label=formatted_solver, marker='o', capsize=5, linewidth=2, markersize=6, color=colors[i % len(colors)])

    # Get system info for title
    system_info = df.iloc[0]
    system_title = f"{system_info['os']} | {system_info['cpu_model']} | Python {system_info['python_version']}"

    ax.set_xlabel('Problem Dimensions', fontsize=12)
    ax.set_ylabel('Average Energy Consumed (kWh)', fontsize=12)
    ax.set_title(f'Energy Consumption vs Problem Dimensions\n{system_title}', fontsize=14, pad=20)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Set x-axis to show dimensions
    unique_dims = df['n_dims'].unique()
    ax.set_xticks(sorted(unique_dims))
    plt.tight_layout()
    plt.close()

def plot_relative_energy_heatmap(df, ax=None):
    """
    Plot relative energy performance as a heatmap.

    :param df: DataFrame containing energy data minimally with columns
        ['solver', 'problem', 'n_dims', 'energy_consumed', 'os', 'cpu_model',
        'python_version'].
    :param ax: The matplotlib axes object to plot on
    """
    if df.empty or ax is None:
        warnings.warn("Dataframe is empty or axes object is None for plot_relative_energy_heatmap.")
        return

    # Check if required columns exist
    required_cols = ['solver', 'problem', 'n_dims', 'energy_consumed', 'os', 'cpu_model', 'python_version']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        warnings.warn(f"Dataframe missing required columns for plot_relative_energy_heatmap: {missing_cols}")
        return

    df = df.copy()

    # Calculate mean energy for each solver-problem-dimension combination
    grouped = df.groupby(['solver', 'problem', 'n_dims'])['energy_consumed'].mean().reset_index()

    # Create pivot table for comparison
    comparison_data = []
    for problem in grouped['problem'].unique():
        for dims in grouped['n_dims'].unique():
            subset = grouped[(grouped['problem'] == problem) & (grouped['n_dims'] == dims)]
            if len(subset) > 1:  # Only compare if multiple solvers ran this configuration
                min_energy = subset['energy_consumed'].min()
                config_label = f"{problem} ({dims}d)"

                for _, solver_row in subset.iterrows():
                    solver_name = str(solver_row['solver']).replace('_', ' ').title()
                    relative_energy = solver_row['energy_consumed'] / min_energy
                    comparison_data.append({
                        'Configuration': config_label,
                        'Solver': solver_name,
                        'Relative Energy': relative_energy
                    })

    if not comparison_data:
        return

    comp_df = pd.DataFrame(comparison_data)

    # Create pivot table for heatmap
    pivot_table = comp_df.pivot(index='Configuration', columns='Solver', values='Relative Energy')

    # Create the heatmap using imshow
    im = ax.imshow(pivot_table.values, cmap='RdYlGn_r', aspect='auto', vmin=1.0)

    # Set ticks and labels
    ax.set_xticks(range(len(pivot_table.columns)))
    ax.set_xticklabels(pivot_table.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(pivot_table.index)))
    ax.set_yticklabels(pivot_table.index)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Relative Energy Usage (1.0 = best)', rotation=270, labelpad=20)

    # Add text annotations
    for i in range(len(pivot_table.index)):
        for j in range(len(pivot_table.columns)):
            if not pd.isna(pivot_table.iloc[i, j]):
                # Choose text color based on background
                text_color = 'white' if pivot_table.iloc[i, j] > 1.5 else 'black'
                ax.text(j, i, f'{pivot_table.iloc[i, j]:.2f}',
                       ha="center", va="center", color=text_color, fontweight='bold')

    # Get system info for title
    system_info = df.iloc[0]
    system_title = f"{system_info['os']} | {system_info['cpu_model']} | Python {system_info['python_version']}"

    ax.set_xlabel('Solver', fontsize=12)
    ax.set_ylabel('Problem Configuration', fontsize=12)
    ax.set_title(f'Relative Energy Performance Heatmap\n{system_title}', fontsize=14, pad=20)
    plt.tight_layout()
    plt.close()

def plot_system_specs(df, ax=None):
    """
    Plot system specifications information.

    :param df: DataFrame containing energy data minimally with columns
        ['os', 'python_version', 'codecarbon_version', 'cpu_model',
        'cpu_count', 'gpu_model', 'gpu_count', 'ram_total_size'].
    :param ax: The matplotlib axes object to plot on
    """
    if df.empty or ax is None:
        warnings.warn("Dataframe is empty or axes object is None for plot_system_specs.")
        return

    # Check if required columns exist
    required_cols = ['os', 'python_version', 'codecarbon_version', 'cpu_model', 'cpu_count', 'gpu_model', 'gpu_count', 'ram_total_size']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        warnings.warn(f"Dataframe missing required columns for plot_system_specs: {missing_cols}")
        return

    df = df.copy()

    ax.axis('off')

    system_info = df.iloc[0]
    specs_text = f"""System Specifications:

    Operating System: {system_info['os']}
    Python Version: {system_info['python_version']}
    CodeCarbon Version: {system_info['codecarbon_version']}

    CPU Model: {system_info['cpu_model']}
    CPU Count: {system_info['cpu_count']}

    GPU Model: {system_info['gpu_model'] if pd.notna(system_info['gpu_model']) else 'None detected'}
    GPU Count: {system_info['gpu_count'] if pd.notna(system_info['gpu_count']) else 0}

    Total RAM: {system_info['ram_total_size']} GB"""

    ax.text(0.5, 0.5, specs_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='center', horizontalalignment='center',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

    ax.set_title('Benchmark System Information', fontsize=16, pad=20)
    plt.tight_layout()
    plt.close()
