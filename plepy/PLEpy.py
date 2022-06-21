import json
import copy

import numpy as np
import pyomo.environ as pe

from plepy.helper import recur_to_json


class PLEpy:

    def __init__(self, model, pnames: list, indices=None, solver="ipopt",
                 solver_kwds={}, tee=False, dae=None, dae_kwds={},
                 presolve=False):
        """Profile Likelihood Estimator object

        Args
        ----
        model : Pyomo model
        pnames : list
            names of estimated parameters in model

        Keywords
        --------
        indices : dict, optional
            dictionary of indices for estimated parameters of format:
            {"index name": values}, "index name" does not need to be
            the name of an index in the model, by default None
        solver : str, optional
            name of solver for Pyomo to use, by default "ipopt"
        solver_kwds : dict, optional

        tee : bool, optional
            print Pyomo iterations at each step, by default False
        dae : discretization method for dae package, optional
            "finite_difference", "collocation", or None, by default
            None
        dae_kwds : dict, optional
            keywords for dae package, by default {}
        presolve : bool, optional
            if True, model needs to be solved first, by default False
        """
        # Define solver & options
        solver_opts = {
            "linear_solver": "ma27",
            "tol": 1e-6
        }
        solver_opts = {**solver_opts, **solver_kwds}
        opt = pe.SolverFactory(solver)
        opt.options = solver_opts
        self.solver = opt
        self.tee = tee

        self.m = model
        # Discretize and solve model if necessary
        if dae and presolve:
            assert isinstance(dae, str)
            tfd = pe.TransformationFactory("dae." + dae)
            tfd.apply_to(self.m, **dae_kwds)
        if presolve:
            r = self.solver.solve(self.m, tee=self.tee)
            self.m.solutions.load_from(r)

        # Gather parameters to be profiled, their optimized values, and
        # bounds list of names of parameters to be profiled
        self.pnames = pnames
        self.indices = indices
        obj_tuple = next(self.m.component_map(ctype=pe.Objective).items())
        self.objname = obj_tuple[0]
        self.objitem = obj_tuple[1]
        self.obj = pe.value(m_obj)    # original objective value
        pprofile = {p: self.m.find_component(p) for p in self.pnames}
        # list of Pyomo Variable objects to be profiled
        self.plist = pprofile
        # determine which variables are indexed
        self.pindexed = {p: self.plist[p].is_indexed() for p in self.pnames}
        # make empty dictionaries for optimal parameters and their bounds
        self.pidx = {}
        self.popt = {}
        self.pbounds = {}
        for p in self.pnames:
            # for indexed parameters...
            if not self.pindexed[p]:
                # get optimal solution
                self.popt[p] = pe.value(self.plist[p])
                # get parameter bounds
                self.pbounds[p] = self.plist[p].bounds

    def set_index(self, pname: str, *args):
        """Indicate which index to use for parameter, pname

        Args
        ----
        pname : str
            name of parameter to set index of
        *args : str
            string specifying the name of the index or indices to use
            for parameter
        """
        import itertools as it

        assert self.pindexed[pname]
        for arg in args:
            assert arg in self.indices.keys()
        # get list of indices in same order as *args
        pindex = list(it.product(*[self.indices[arg] for arg in args]))
        self.pidx[pname] = pindex
        self.popt[pname] = {}
        self.pbounds[pname] = {}
        for k in pindex:
            # get optimal solutions
            self.popt[pname][k] = pe.value(self.plist[pname][k])
            # get parameter bounds
            self.pbounds[pname][k] = self.plist[pname][k].bounds

    def getval(self, pname: str):
        if self.pindexed[pname]:
            return {k: pe.value(self.plist[pname][k])
                    for k in self.pidx[pname]}
        else:
            return pe.value(self.plist[pname])

    def setval(self, pname: str, val):
        if self.pindexed[pname]:
            self.plist[pname].set_values(val)
        else:
            self.plist[pname].set_value(val)

    def get_clevel(self, alpha: float=0.05, sse_func=None, sse_args=[]):
        # determine confidence threshold value
        from scipy.stats.distributions import chi2
        etol = chi2.isf(alpha, df=1)
        if sse_func is not None:
            obj = sse_func(self.m, *sse_args)
        else:
            obj = self.obj
        clevel = etol/2 + np.log(obj)
        self.clevel = clevel
        return clevel

    def get_PL(self, pnames="all", n: int=20, min_step: float=1e-3,
               dtol: float=0.2, save: bool=False, fname="tmp_PLfile.json",
               debug=False, sse_func=None, sse_args=[]):
        """Once bounds are found, calculate likelihood profiles for
        each parameter

        Args
        ----
        pnames: list or str
            name(s) of parameters to generate likelihood profiles for,
            or "all" to generate profiles for all model parameters, by
            default "all"

        Keywords
        --------
        n : int, optional
            minimum number of discretization points between optimum and
            each parameter bound, by default 20
        min_step : float, optional
            minimum allowable difference between two discretization
            points, by default 1e-3
        dtol : float, optional
            maximum error change between two points, by default 0.2
        save: bool, optional
            if True, will save results to a JSON file, by default False
        fname: str or path, optional
            location to save JSON file (if save=True),
            by default "tmp_PLfile.json"
        debug: bool, optional
            whether to print additional information during profiling, by
            default False
        sse_func: Callable, optional
            function handle that calculates the sum of squared error if using
            something other than model's objective function, by default None
        sse_args: list, optional
            list of arguments to pass to sse_func
        """
        from plepy.helper import sig_figs

        def inner_loop(xopt, xb, direct=1, idx=None, debug=False,
                       sse_func=None, sse_args=[]) -> dict:
            from plepy.helper import sflag, sig_figs

            pdict = {}
            if direct:
                print("Going up...")
                x0 = np.linspace(xopt, xb, n+2, endpoint=True)
            else:
                print("Going down...")
                x0 = np.linspace(xb, xopt, n+2, endpoint=True)
            if debug:
                sfs = [f"{sig_figs(xi, 3):0<4}" for xi in x0]
                xstr = ", ".join([f"{sf:>5}" for sf in sfs])
                print("Evaluating at:".ljust(17), f"[{xstr}]")
            # evaluate objective at each discretization point
            for w, x in enumerate(x0):
                xdict = {}
                if w == 0:
                    for p in self.pnames:
                        self.setval(p, self.popt[p])
                else:
                    for p in self.pnames:
                        prevx = pdict[x0[w-1]][p]
                        self.setval(p, prevx)
                try:
                    rx = self.m_eval(pname, x, idx=idx, reset=False)
                    xdict["flag"] = sflag(rx)
                    self.m.solutions.load_from(rx)
                    if sse_func is not None:
                        obj = sse_func(self.m, *sse_args)
                    else:
                        obj = pe.value(self.objitem)
                    xdict["obj"] = np.log(obj)
                    # store values of other parameters at each point
                    for p in self.pnames:
                        xdict[p] = self.getval(p)
                except ValueError:
                    xdict = copy.deepcopy(pdict[x0[w-1]])
                pdict[x] = xdict
            if direct:
                x_out = x0[1:]
                x_in = x0[:-1]
            else:
                x_out = x0[:-1]
                x_in = x0[1:]
            # calculate magnitude of step sizes
            dx = x_out - x_in
            y0 = np.array([pdict[x]["obj"] for x in x0])
            if debug:
                sfs = [f"{sig_figs(yi, 3):0<4}" for yi in y0]
                ystr = ", ".join([f"{sf:>5}" for sf in sfs])
                print("Objective values:".ljust(17), f"[{ystr}]")
            if direct:
                y_out = y0[1:]
                y_in = y0[:-1]
            else:
                y_out = y0[:-1]
                y_in = y0[1:]
            # calculate magnitude of objective value changes between
            # each step
            dy = np.abs(y_out - y_in)
            # pull indices where objective value change is greater than
            # threshold value (dtol) and step size is greater than
            # minimum
            ierr = [(i > dtol and j > min_step)
                             for i, j in zip(dy, dx)]
            if debug:
                bstr = ", ".join([f"{b!s:>5}" for b in ierr])
                print("Discretize again:".ljust(17), f"[{bstr}]")
            itr = 0
            # For intervals of large change (above dtol), calculate
            # values at midpoint. Repeat until no large changes or
            # minimum step size is reached.
            while len(ierr) != 0:
                print(f"iter: {itr}")
                x_oerr = np.array([j for i, j in zip(ierr, x_out) if i])
                x_ierr = np.array([j for i, j in zip(ierr, x_in) if i])
                x_mid = 0.5*(x_oerr + x_ierr)
                for w, x in enumerate(x_mid):
                    xdict = {}
                    for p in self.pnames:
                        prevx = pdict[x_ierr[w]][p]
                        self.setval(p, prevx)
                    try:
                        rx = self.m_eval(pname, x, idx=idx, reset=False)
                        xdict["flag"] = sflag(rx)
                        self.m.solutions.load_from(rx)
                        if sse_func is not None:
                            obj = sse_func(self.m, *sse_args)
                        else:
                            obj = pe.value(self.objitem)
                        xdict["obj"] = np.log(obj)
                        # store values of other parameters at each pt
                        for p in self.pnames:
                            xdict[p] = self.getval(p)
                    except ValueError:
                        xdict = copy.deepcopy(pdict[x_ierr[w]])
                    pdict[x] = xdict
                # get parameter values needed to calculate change in
                # error over intervals that have not converged
                x0 = np.array(sorted(set([*x_oerr, *x_mid, *x_ierr])))
                if debug:
                    sfs = [f"{sig_figs(xi, 3):0<4}" for xi in x0]
                    xstr = ", ".join([f"{sf:>5}" for sf in sfs])
                    print("Evaluating at:".ljust(17), f"[{xstr}]")
                x_out = x0[1:]
                x_in = x0[:-1]
                # calculate magnitude of step sizes
                dx = x_out - x_in
                y0 = np.array([pdict[x]["obj"] for x in x0])
                if debug:
                    sfs = [f"{sig_figs(yi, 3):0<4}" for yi in y0]
                    ystr = ", ".join([f"{sf:>5}" for sf in sfs])
                    print("Objective values:".ljust(17), f"[{ystr}]")
                y_out = y0[1:]
                y_in = y0[:-1]
                # calculate magnitude of objective value change between
                # each step
                dy = np.abs(y_out - y_in)
                # pull indices where objective value change is greater
                # than threshold value (dtol) and step size is greater
                # than minimum
                ierr = [(i > dtol and j > min_step)
                                 for i, j in zip(dy, dx)]
                if debug:
                    bstr = ", ".join([f"{b!s:>5}" for b in ierr])
                    print("Discretize again:".ljust(17), f"[{bstr}]")
                itr += 1
            return pdict

        if isinstance(pnames, str):
            if pnames == "all":
                pnames = list(self.pnames)
            else:
                pnames = [pnames]

        # master dictionary for all parameter likelihood profiles
        PLdict = {}
        # generate profiles for parameters indicated
        for pname in pnames:
            print(f"Profiling {pname}...")
            # make sure upper and lower confidence limits have been
            # specified or calculated using get_clims()
            emsg = ("Parameter confidence limits must be determined "
                    "prior to calculating likelihood profile.\nTry "
                    "running .get_clims() method first.")
            assert self.parlb[pname] is not None, emsg
            assert self.parub[pname] is not None, emsg

            if self.pindexed[pname]:
                parPL = {}
                for k in self.pidx[pname]:
                    self.plist[pname][k].fix()
                    xopt = self.popt[pname][k]
                    xlb = self.parlb[pname][k]
                    xub = self.parub[pname][k]
                    print(f"Index: {k}")
                    print(f"Optimized value: {sig_figs(xopt, 3)}",
                          f"Lower C.L.: {sig_figs(xlb, 3)}",
                          f"Upper C.L.: {sig_figs(xub, 3)}", sep="\n")
                    kPLup = inner_loop(xopt, xub, direct=1, idx=k, debug=debug,
                                       sse_func=sse_func, sse_args=sse_args)
                    kPLdn = inner_loop(xopt, xlb, direct=0, idx=k, debug=debug,
                                       sse_func=sse_func, sse_args=sse_args)
                    kPL = {**kPLup, **kPLdn}
                    parPL[k] = kPL
                    self.plist[pname][k].free()
                PLdict[pname] = parPL
            else:
                self.plist[pname].fix()
                xopt = self.popt[pname]
                xlb = self.parlb[pname]
                xub = self.parub[pname]
                # discretize each half separately
                print(f"Optimized value: {sig_figs(xopt, 3)}",
                      f"Lower C.L.: {sig_figs(xlb, 3)}",
                      f"Upper C.L.: {sig_figs(xub, 3)}", sep="\n")
                parPLup = inner_loop(xopt, xub, direct=1, debug=debug,
                                     sse_func=sse_func, sse_args=sse_args)
                parPLdn = inner_loop(xopt, xlb, direct=0, debug=debug,
                                     sse_func=sse_func, sse_args=sse_args)
                # combine results into parameter profile likelihood
                parPL = {**parPLup, **parPLdn}
                PLdict[pname] = parPL
                self.plist[pname].free()
        self.PLdict = PLdict
        if save:
            jdict = recur_to_json(PLdict)
            with open(fname, "w") as f:
                json.dump(jdict, f)

    def plot_PL(self, **kwds):
        from plepy.helper import plot_PL

        assert isinstance(self.PLdict, dict)
        assert isinstance(self.clevel, float)
        jdict = recur_to_json(self.PLdict)
        figs, axs = plot_PL(jdict, self.clevel, **kwds)
        return figs, axs

    def m_eval(self, pname: str, pardr, idx=None, reset=True):
        # initialize all parameters at their optimal value (to ensure
        # feasibility)
        if reset:
            for p in self.pnames:
                self.setval(p, self.popt[p])
        # if parameter is indexed, set value of parameter at specified
        # index to pardr
        if idx is not None:
            self.plist[pname][idx].set_value(pardr)
        # if parameter is unindexed, set value of parameter to pardr
        else:
            self.plist[pname].set_value(pardr)
        # evalutate model at this point
        return self.solver.solve(self.m, tee=self.tee)

    def bsearch(self, pname: str, clevel: float, acc: float,
                direct: int=1, idx=None, sse_func=None, sse_args=[]) -> float:
        """Binary search for confidence limit
        Args
        ----
        pname : str
            parameter name
        clevel: float
            value of log of objective function at confidence limit
        acc: float
            maximum fractional difference between binary search bounds
            allowed for convergence

        Keywords
        --------
        direct : int, optional
            direction to search (0=downwards, 1=upwards), by default 1
        idx: optional
            for indexed parameters, the value of the index to get the
            confidence limits for, by default None
        sse_func: Callable, optional
            function handle that calculates the sum of squared error if using
            something other than model's objective function, by default None
        sse_args: list, optional
            list of arguments to pass to sse_func

        Returns
        -------
        pCI
            value of parameter bound
        """
        import math
        from plepy.helper import sflag, sig_figs

        def feasible_bound(pname, x_mid, x_in, r_mid, clevel, ctol, direct=1,
                           idx=None, nsig=3, sse_func=None, sse_args=[]):
            # Find furthest feasible bound
            print("Entering feasibility check...")
            print(" "*80)
            print("iter".center(10), "high".center(10), "low".center(10),
                sep=" | ")
            print("-"*10, "-"*10, "-"*10, sep="-+-")

            self.m.solutions.load_from(r_mid)
            if sse_func is not None:
                obj = sse_func(self.m, *sse_args)
            else:
                obj = pe.value(self.objitem)
            err = np.log(obj)
            if direct:
                x_high = x_mid
                x_low = x_in
            else:
                x_high = x_in
                x_low = x_mid
            fiter = 0
            while (fcheck == 1 or err < clevel) and x_range > ctol:
                hstr = f"{sig_figs(x_high, nsig)}"
                lstr = f"{sig_figs(x_low, nsig)}"
                print(f"{fiter:^10}", f"{hstr:>10}", f"{lstr:>10}", sep=" | ")
                # check convergence criteria
                x_range = x_high - x_low
                ctol = x_high*acc
                # evaluate at midpoint
                x_mid = 0.5*(x_high + x_low)
                r_mid = self.m_eval(pname, x_mid, idx)
                fcheck = sflag(r_mid)
                # if infeasible, continue search inward from current
                # midpoint
                if fcheck == 1:
                    x_out = float(x_mid)
                self.m.solutions.load_from(r_mid)
                if sse_func is not None:
                    obj = sse_func(self.m, *sse_args)
                else:
                    obj = pe.value(self.objitem)
                err = np.log(obj)
                # if feasbile, but not over CL threshold, continue
                # search outward from current midpoint
                if fcheck == 0 and err < clevel:
                    x_in = float(x_mid)
                if direct:
                    x_high = x_out
                    x_low = x_in
                else:
                    x_high = x_in
                    x_low = x_out
                fiter += 1
            x_out = float(x_mid)
            print(" "*80)
            print(f"Feasibile limit: {sig_figs(x_out, nsig)}")
            print(" "*80)
            r_mid = self.m_eval(pname, x_mid, idx=idx)
            fcheck = sflag(r_mid)
            if fcheck == 0:
                print("Continuing with binary search...")
                print(" "*80)
            return x_out, r_mid, fcheck

        # Number of sig. figs to print (based on acc)
        nsig = int(-math.log10(acc) + 2)
        # manually change parameter of interest
        if idx is None:
            self.plist[pname].fix()
            x_out = float(self.pbounds[pname][direct])
            x_in = float(self.popt[pname])
        else:
            self.plist[pname][idx].fix()
            x_out = float(self.pbounds[pname][idx][direct])
            x_in = float(self.popt[pname][idx])

        # Initialize values based on direction
        x_mid = x_out
        # for upper CI search
        if direct:
            x_high = x_out
            x_low = x_in
            plc = "upper"
            puc = "Upper"
            no_lim = float(x_out)
        # for lower CI search
        else:
            x_high = x_in
            x_low = x_out
            plc = "lower"
            puc = "Lower"
            no_lim = float(x_out)

        # Print search info
        print(" "*80)
        print(f"Parameter: {pname}")
        if idx is not None:
            print(f"Index: {idx}")
        print(f"Bound: {puc}")
        print(" "*80)

        # check convergence criteria
        x_range = x_high - x_low
        ctol = x_high*acc

        # Find outermost feasible value
        # evaluate at outer bound
        r_mid = self.m_eval(pname, x_mid, idx=idx)
        fcheck = sflag(r_mid)
        # If solution is infeasible, trigger search for feasible bound
        if fcheck == 1:
            x_out, r_mid, fcheck = feasible_bound(pname, x_mid, x_in, r_mid,
                                                  clevel, ctol, direct=direct,
                                                  idx=idx)
        self.m.solutions.load_from(r_mid)
        if sse_func is not None:
            obj = sse_func(self.m, *sse_args)
        else:
            obj = pe.value(self.objitem)
        err = np.log(obj)
        # If solution is *still* infeasible, there is no feasible upper limit.
        # Set to xopt
        if fcheck == 1:
            pCI = x_out
            print(f"No feasible {plc} CI! Setting to optimum value.")
            print(f"Error at bound: {sig_figs(err, nsig)}")
            print(f"Confidence threshold: {sig_figs(clevel, nsig)}")
        # If solution is feasible and the error is less than the value
        # at the confidence limit, there is no CI in that direction.
        # Set to bound.
        elif fcheck == 0 and err < clevel:
            pCI = x_out
            print(f"No {plc} CI! Setting to {plc} bound.")
            print(f"Error at bound: {sig_figs(err, nsig)}")
            print(f"Confidence threshold: {sig_figs(clevel, nsig)}")
        else:
            if direct:
                x_high = x_out
                x_low = x_in
            else:
                x_high = x_in
                x_low = x_out
            biter = 0
            # repeat until convergence criteria is met
            # (i.e. x_high - x_low < x_high*acc)
            print(" "*80)
            print("iter".center(10), "high".center(10), "low".center(10),
                sep=" | ")
            print("-"*10, "-"*10, "-"*10, sep="-+-")
            while x_range > ctol:
                hstr = f"{sig_figs(x_high, nsig)}"
                lstr = f"{sig_figs(x_low, nsig)}"
                print(f"{biter:^10}", f"{hstr:>10}", f"{lstr:>10}",
                        sep=" | ")
                # check convergence criteria
                x_range = x_high - x_low
                ctol = x_high*acc
                # evaluate at midpoint
                x_mid = 0.5*(x_high + x_low)
                r_mid = self.m_eval(pname, x_mid, idx=idx)
                fcheck = sflag(r_mid)
                self.m.solutions.load_from(r_mid)
                if sse_func is not None:
                    obj = sse_func(self.m, *sse_args)
                else:
                    obj = pe.value(self.objitem)
                err = np.log(obj)
                biter += 1
                # if midpoint infeasible, continue search inward
                if fcheck == 1:
                    x_out = float(x_mid)
                # if midpoint over CL, continue search inward
                elif err > clevel:
                    x_out = float(x_mid)
                # if midpoint under CL, continue search outward
                else:
                    x_in = float(x_mid)
                if direct:
                    x_high = x_out
                    x_low = x_in
                else:
                    x_high = x_in
                    x_low = x_out
            pCI = x_mid
            print(" "*80)
            print(f"{puc} CI of {sig_figs(pCI, nsig)} found!")
        # reset parameter
        self.setval(pname, self.popt[pname])
        if idx is None:
            self.plist[pname].free()
        else:
            self.plist[pname][idx].free()
        return pCI

    def get_clims(self, pnames="all", idx=None, alpha: float=0.05,
                  acc: float=0.01, sse_func=None, sse_args=[]):
        """Get confidence limits of parameters
        Keywords
        --------
        pnames: list or str, optional
            name of parameter(s) to get confidence limits for, if "all"
            will find limits for all parameters, by default "all"
        idx: optional
            index of parameter to get confidence limits for - pnames must be
            a single parameter, by default None
        alpha : float, optional
            confidence level, by default 0.05
        acc : float, optional
            maximum fractional difference between binary search bounds
            allowed for convergence, by default 0.01
        sse_func: Callable, optional
            function handle that calculates the sum of squared error if using
            something other than model's objective function, by default None
        sse_args: list, optional
            list of arguments to pass to sse_func
        """
        if isinstance(pnames, str):
            if pnames == "all":
                assert idx is None
                pnames = list(self.pnames)
            else:
                if idx is not None:
                    assert idx in self.pidx[pnames]
                pnames = [pnames]
        else:
            assert idx is None

        # Define threshold of confidence level
        clevel = self.get_clevel(alpha, sse_func=sse_func, sse_args=sse_args)

        # create dictionaries for the confidence limits with the same
        # structure as self.popt
        if hasattr(self, "parub"):
            parub = copy.deepcopy(dict(self.parub))
        else:
            parub = copy.deepcopy(dict(self.popt))
        if hasattr(self, "parlb"):
            parlb = copy.deepcopy(dict(self.parlb))
        else:
            parlb = copy.deepcopy(dict(self.popt))
        # Get upper & lower confidence limits
        for pname in pnames:
            # for indexed variables
            if self.pindexed[pname]:
                if idx is not None:
                    parlb[pname][idx] = self.bsearch(pname, clevel, acc,
                                                     direct=0, idx=idx,
                                                     sse_func=sse_func,
                                                     sse_args=sse_args)
                    parub[pname][idx] = self.bsearch(pname, clevel, acc,
                                                     direct=1, idx=idx,
                                                     sse_func=sse_func,
                                                     sse_args=sse_args)
                else:
                    for idx in self.pidx[pname]:
                        parlb[pname][idx] = self.bsearch(pname, clevel, acc,
                                                         direct=0, idx=idx,
                                                         sse_func=sse_func,
                                                         sse_args=sse_args)
                        parub[pname][idx] = self.bsearch(pname, clevel, acc,
                                                         direct=1, idx=idx,
                                                         sse_func=sse_func,
                                                         sse_args=sse_args)
            # for unindexed variables
            else:
                parlb[pname] = self.bsearch(pname, clevel, acc, direct=0,
                                            sse_func=sse_func,
                                            sse_args=sse_args)
                parub[pname] = self.bsearch(pname, clevel, acc, direct=1,
                                            sse_func=sse_func,
                                            sse_args=sse_args)
        self.parub = parub
        self.parlb = parlb

    def to_json(self, filename):
        # save existing attributes to JSON file
        atts = ["pnames", "indices", "obj", "pindexed", "pidx", "popt",
                "pbounds", "parlb", "parub", "clevel", "PLdict"]
        sv_dict = {}
        for att in atts:
            # if PLEpy attribute exists, convert it to a JSON
            # compatible form and add it to sv_dict
            try:
                sv_var = getattr(self, att)
                if isinstance(sv_var, dict):
                    sv_var = recur_to_json(sv_var)
                sv_dict[att] = sv_var
            except AttributeError:
                print(f"Attribute '{att}' does not exist. Skipping.")
        # write to JSON file
        with open(filename, "w") as f:
            json.dump(sv_dict, f)

    def load_json(self, filename):
        from plepy.helper import recur_load_json
        # load PL data from a json file
        atts = ["pidx", "popt", "pbounds", "parlb", "parub", "clevel",
                "PLdict"]
        with open(filename, "r") as f:
            sv_dict = json.load(f)
        for att in atts:
            # check for each PLEpy attribute and unserialize it from
            # JSON format
            try:
                sv_var = sv_dict[att]
                if att == "pidx":
                    sv_var = {k: [tuple(i) for i in sv_var[k]]
                              for k in sv_var.keys()}
                elif att == "clevel":
                    pass
                else:
                    sv_var = recur_load_json(sv_var)
                setattr(self, att, sv_var)
            except KeyError:
                print(f"Attribute '{att}' not yet defined.")
