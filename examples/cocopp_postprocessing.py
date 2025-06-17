import cocopp
import scipy.optimize as opt
import pygold.benchmark_functions as bf
import pygold.benchmark_tools as bt

problems = bf.__nD__[::3]

solvers = [opt.shgo, opt.dual_annealing]

bt.run_solvers(solvers, problems, test_dimensions=[1,2,3,4,5,6], n_iters= 2, verbose=True)

bt.configure_testbed(problems, test_dimensions=[1,2,3,4,5,6])

# Filter warnings about unknown cocopp suite
cocopp.main(["output_data/shgo", "output_data/dual_annealing"])
