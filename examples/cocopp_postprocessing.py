import cocopp
import scipy.optimize as opt
import pygold
import pygold.problems.standard_problems as bp

# Select test problems
# We'll use all the standard nD problems
problems = bp.__nD__

# Select solvers to test
# We'll use SHGO and dual_annealing from scipy.optimize
solvers = [opt.shgo, opt.dual_annealing]

# Run the solvers on the problems
# We'll use 2 iterations and test dimensions from 1 to 6
# run_solvers will run the solvers on the problems and save the results
# in the output_data folder in the format expected by COCOPP
pygold.run_solvers(solvers, problems, test_dimensions=[1,2,3,4,5,6], n_iters= 2, verbose=True)

# Configure the testbed for COOCOPP
# This will set up the postprocessing environment for COCOPP
pygold.configure_testbed(problems, test_dimensions=[1,2,3,4,5,6])

# Run the postprocessing using COCOPP
cocopp.main(["output_data/shgo", "output_data/dual_annealing"])
