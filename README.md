# Global Optimization Benchmark

A Python package for benchmarking global optimization algorithms using a suite of standard test functions and performance profiling tools.

## Features
- A large collection of benchmark functions (multimodal, unimodal, scalable, etc.)
- Tools for running and evaluating solvers on benchmark problems
- Performance ratio and performance profile computation
- Example scripts for running and comparing solvers

## Installation

Install using [TOOL]

```bash
pip install 
```

## Usage

```python
import global_opt_bench.benchmark_functions as bf
import global_opt_bench.benchmark_tools as bt

# List all available benchmark functions
print(bf.__All__)

# Run solvers on selected problems
solvers = [...]  # List of solver callables
problems = bf.__All__[:3]  # Select a few problems
results = bt.run_solvers_time(solvers, problems)

# Compute and plot performance profiles
ratios = bt.compute_performance_ratios(results)
profiles = bt.compute_performance_profiles(ratios)
bt.plot_performance_profiles(profiles)
```

## Documentation

Full documentation is available at [LINK].

## Contributing

Contributions are welcome! Please open issues or pull requests for bug fixes, new features, or improvements.

## License

This project is licensed under the [LICENSE] License.