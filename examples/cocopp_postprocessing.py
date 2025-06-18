import cocopp
import scipy.optimize as opt
import pygold
import pygold.problems.standard_problems as bp

problems = bp.__nD__[::3]

solvers = [opt.shgo, opt.dual_annealing]

pygold.run_solvers(solvers, problems, test_dimensions=[1,2,3,4,5,6], n_iters= 2, verbose=True)

pygold.configure_testbed(problems, test_dimensions=[1,2,3,4,5,6])

# Filter warnings about unknown cocopp suite
cocopp.main(["output_data/shgo", "output_data/dual_annealing"])
