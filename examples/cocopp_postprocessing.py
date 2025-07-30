import cocopp
import scipy.optimize as opt
import pygold

# 1. Select test problems
# We'll use all the standard nD problems
# Note: COCOPP only supports postprocessing for nD problems
# To test with functions in specific dimensions, use the functions provided
# in the pygold.postprocessing module
problems = pygold.get_standard_problems("nD")

# 2. Select solvers to test
# We'll use differential_evolution and dual_annealing from scipy.optimize
solvers = [opt.differential_evolution, opt.dual_annealing]

# 3. Run the solvers on the problems
# We'll use 2 iterations and test dimensions from 2 to 10
# test_dimensions must have exactly 6 dimensions, more or less can cause
# unexpected behavior like repeated, missing or incorrect graphs (see the docs)
# run_solvers will run the solvers on the problems and save the results
# in the output_data folder in the format expected by COCOPP
# COCOPP does not support energy metrics, so we won't track energy here
pygold.run_solvers(solvers, problems, test_dimensions=[2, 4, 5, 7, 8, 10], n_iters=2, verbose=True, track_energy=False)

# 4. Configure the testbed for COOCOPP
# This will set up the postprocessing environment for COCOPP
# The test_dimensions must match the ones used in run_solvers and have exactly
# 6 dimensions
pygold.configure_testbed(problems, test_dimensions=[2, 4, 5, 7, 8, 10], n_solvers=2)

# 5. Run the postprocessing using COCOPP
cocopp.main(["output_data/differential_evolution", "output_data/dual_annealing"])
