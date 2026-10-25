import numpy as np
import pytest

from nabia_python_bridge import optimization as opt


def rosenbrock(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def rosenbrock_grad(x):
    return np.array([-2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2), 200 * (x[1] - x[0] ** 2)])


def test_lbfgs_both_callback_styles():
    p = opt.LBFGSParams(dim=2, tolerance=1e-10, max_iter=2000)
    a = opt.LBFGS(p, rosenbrock, rosenbrock_grad)
    assert a.optimize(np.array([-1.2, 1.0])) == pytest.approx([1.0, 1.0], abs=1e-6)
    assert a.result.converged and a.result.status == opt.LBFGSStatus.Converged
    b = opt.LBFGS(p, lambda x: (rosenbrock(x), rosenbrock_grad(x)))
    assert b.optimize(np.array([-1.2, 1.0])) == pytest.approx([1.0, 1.0], abs=1e-6)


def test_callback_errors_propagate():
    def broken(x):
        raise KeyError("boom")

    with pytest.raises(KeyError):
        opt.LBFGS(opt.LBFGSParams(dim=2), broken, rosenbrock_grad).optimize()
    with pytest.raises(ValueError):  # gradient of the wrong size
        opt.LBFGS(opt.LBFGSParams(dim=2), rosenbrock, lambda x: np.zeros(3)).optimize()


def quadratic(x):
    return float(((x - np.array([2.0, 2.0])) ** 2).sum())


def quadratic_grad(x):
    return 2 * (x - np.array([2.0, 2.0]))


@pytest.mark.parametrize("solver", [opt.AugmentedLagrangian, opt.SQP])
def test_constrained_solvers(solver):
    # min |x - (2, 2)|^2  s.t.  x0 + x1 = 1, 0 <= x <= 1  ->  (0.5, 0.5)
    lin = opt.LinearConstraint(A_eq=np.ones((1, 2)), b_eq=np.ones(1),
                               A_ineq=np.vstack([np.eye(2), -np.eye(2)]), b_ineq=np.array([1.0, 1.0, 0.0, 0.0]))
    p = opt.ConvexOptimizationParams(dim=2, tolerance=1e-9, max_iter=200, constraints=lin)
    s = solver(p, quadratic, quadratic_grad)
    assert s.optimize() == pytest.approx([0.5, 0.5], abs=1e-6)
    assert s.result.converged
    assert lin.is_satisfied(s.result.x, 1e-6)


def test_box_constraints_as_linear():
    box = opt.BoxConstraints([0.0, -1.0], [1.0, 1.0])
    lin = opt.to_linear_constraint(box, 2)
    assert lin.num_ineq == 4
    p = opt.ConvexOptimizationParams(dim=2, tolerance=1e-9, max_iter=200, constraints=box)
    assert opt.SQP(p, quadratic, quadratic_grad).optimize() == pytest.approx([1.0, 1.0], abs=1e-6)


def test_nelder_mead():
    p = opt.NelderMeadParams(dim=2, bounds=opt.BoxConstraints([-5, -5], [5, 5]))
    nm = opt.NelderMead(p, rosenbrock)
    assert nm.optimize(np.array([-1.0, 2.0])) == pytest.approx([1.0, 1.0], abs=1e-4)
    assert nm.result.status == opt.NelderMeadStatus.Converged


@pytest.mark.parametrize("threads", [1, 4])
def test_differential_evolution_threads_call_python_safely(threads):
    p = opt.DifferentialEvolutionParams(dim=2, bounds=opt.BoxConstraints([-5, -5], [5, 5]), max_generations=400,
                                        strategy=opt.DEStrategy.Best1, seed=7, num_threads=threads)
    de = opt.DifferentialEvolution(p, quadratic)
    assert de.optimize() == pytest.approx([2.0, 2.0], abs=1e-3)
    assert de.result.threads_used == threads


def test_differential_evolution_is_reproducible():
    def run(threads):
        p = opt.DifferentialEvolutionParams(dim=2, bounds=opt.BoxConstraints([-5, -5], [5, 5]), max_generations=50,
                                            seed=3, num_threads=threads)
        return opt.DifferentialEvolution(p, rosenbrock).optimize()

    assert np.array_equal(run(1), run(3))
