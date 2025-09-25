# Copyright (c) 2025 Alliance for Sustainable Energy, LLC

import os
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import cocopp
import cocopp.pproc
import cocopp.testbedsettings
import cocopp.pprldistr
import cocopp.ppfig
import cocopp.toolsdivers

class CustomTestbed(cocopp.testbedsettings.GECCOBBOBTestbed):
    # Variables that must be set to match experiment run
    dims = [2, 4, 6, 8, 10, 12] # Must have len == 6
    short_names = {1: "Ackley", 2: "Rothyp"}
    nfxns = 2
    func_cons_groups = OrderedDict({})

    # Past here should be static
    pptable_target_runlengths = [0.5, 1.2, 3, 10, 50]  # used in config for expensive setting
    pptable_targetsOfInterest = (10, 1, 1e-1, 1e-2, 1e-3, 1e-5, 1e-7)
    settings = dict(
        name='pyGOLD',
        short_names=short_names,
        dimensions_to_display=dims,
        goto_dimension=dims[3],  # auto-focus on this dimension in html
        rldDimsOfInterest=[dims[2], dims[4]],
        tabDimsOfInterest=[dims[2], dims[4]],
        hardesttargetlatex='10^{-8}',  # used for ppfigs, pptable and pptables
        ppfigs_ftarget=1e-8,  # to set target runlength in expensive setting, use genericsettings.target_runlength
        ppfig2_ftarget=1e-8,
        ppfigdim_target_values=(10, 1, 1e-1, 1e-2, 1e-3, 1e-5, 1e-8),
        pprldistr_target_values=(10., 1e-1, 1e-4, 1e-8),
        pprldmany_target_values=10 ** np.arange(2, -8.2, -0.2),
        pprldmany_target_range_latex='$10^{[-8..2]}$',
        ppscatter_target_values=np.array(list(np.logspace(-8, 2, 21)) + [3e21]),  # 21 was 46
        rldValsOfInterest=(10, 1e-1, 1e-4, 1e-8),  # possibly changed in config
        ppfvdistr_min_target=1e-8,
        functions_with_legend=(1, nfxns),
        first_function_number=1,
        last_function_number=nfxns,
        reference_values_hash_dimensions=[],
        pptable_ftarget=1e-8,  # value for determining the success ratio in all tables
        pptable_targetsOfInterest=pptable_targetsOfInterest,
        pptablemany_targetsOfInterest=pptable_targetsOfInterest,
        scenario='fixed',
        reference_algorithm_filename='',
        reference_algorithm_displayname='',
        pptable_target_runlengths=pptable_target_runlengths,
        pptables_target_runlengths=pptable_target_runlengths,
        data_format=cocopp.dataformatsettings.BBOBOldDataFormat(),
        number_of_points=5,  # nb of target function values for each decade
        instancesOfInterest=None,  # None: consider all instances

        # Dynamically generate plots_on_main_html_page based on dims
        plots_on_main_html_page=[
            f'pprldmany_{str(d).zfill(2)}D_noiselessall.svg' for d in dims
        ],
    )

    def __init__(self, targetValues):
        for key, val in cocopp.testbedsettings.CustomTestbed.settings.items():
            setattr(self, key, val)
        self.instantiate_attributes(targetValues)

    def filter(self, dsl):
        """
        Overwrite warning filter.
        """
        return dsl

# Register the class in COCOPP's globals so it's available
setattr(cocopp.testbedsettings, 'CustomTestbed', CustomTestbed)
cocopp.testbedsettings.suite_to_testbed['pyGOLD'] = 'CustomTestbed'

## Overwriting these two functions make the function groups work
def custom_getFuncGroups(self):
    if hasattr(cocopp.testbedsettings.current_testbed, 'func_cons_groups'):
        groups = []
        for group_name, ids in cocopp.testbedsettings.current_testbed.func_cons_groups.items():
            if any(i.funcId in ids for i in self):
                groups.append((group_name, group_name))
        return OrderedDict(groups)

cocopp.pproc.DataSetList.getFuncGroups = custom_getFuncGroups

def customDictByFuncGroupSingleObjective(self):
    res = {}
    if hasattr(cocopp.testbedsettings.current_testbed, 'func_cons_groups'):
        for i in self:
            for group_name, ids in cocopp.testbedsettings.current_testbed.func_cons_groups.items():
                if i.funcId in ids:
                    res.setdefault(group_name, cocopp.pproc.DataSetList()).append(i)
    return res

cocopp.pproc.DataSetList.dictByFuncGroupSingleObjective = customDictByFuncGroupSingleObjective

# Custom html file for updated titles, labels, and descriptions
custom_html_path = os.path.abspath("pygold/cocopp_interface/custom_titles.html")
cocopp.genericsettings.latex_commands_for_html = os.path.splitext(custom_html_path)[0]

# Overwrite two functions to remove inaccurate reference data from ECDF plots
def customComp(dsList0, dsList1, targets, isStoringXMax=False, outputdir='', info='default'):
    """Generate figures of ECDF that compare 2 algorithms.

    :param DataSetList dsList0: list of DataSet instances for ALG0
    :param DataSetList dsList1: list of DataSet instances for ALG1
    :param seq targets: target function values to be displayed
    :param bool isStoringXMax: if set to True, the first call
                               :py:func:`beautifyFVD` sets the globals
                               :py:data:`fmax` and :py:data:`maxEvals`
                               and all subsequent calls will use these
                               values as rightmost xlim in the generated
                               figures.
    :param string outputdir: output directory (must exist)
    :param string info: string suffix for output file names.

    """

    if not isinstance(targets, cocopp.pproc.RunlengthBasedTargetValues):
        targets = cocopp.pproc.TargetValues.cast(targets)

    dictdim0 = dsList0.dictByDim()
    dictdim1 = dsList1.dictByDim()
    for d in set(dictdim0.keys()) & set(dictdim1.keys()):
        maxEvalsFactor = max(max(i.mMaxEvals() / d for i in dictdim0[d]),
                             max(i.mMaxEvals() / d for i in dictdim1[d]))
        if isStoringXMax:
            evalfmax = cocopp.pprldistr.evalfmax
        else:
            evalfmax = None
        if not evalfmax:
            evalfmax = maxEvalsFactor ** 1.05

        filename = os.path.join(outputdir, 'pprldistr_%02dD_%s' % (d, info))
        fig = plt.figure()
        for j in range(len(targets)):
            tmp = cocopp.pprldistr.plotRLDistr(dictdim0[d], lambda fun_dim: targets(fun_dim)[j],
                              (targets.label(j)
                               if isinstance(targets,
                                             cocopp.pproc.RunlengthBasedTargetValues)
                               else targets.loglabel(j)),
                              marker=cocopp.genericsettings.line_styles[1]['marker'],
                              **cocopp.pprldistr.rldStyles[j % len(cocopp.pprldistr.rldStyles)])
            plt.setp(tmp[-1], label=None) # Remove automatic legend
            # Mods are added after to prevent them from appearing in the legend
            plt.setp(tmp, markersize=20.,
                     markeredgewidth=plt.getp(tmp[-1], 'linewidth'),
                     markeredgecolor=plt.getp(tmp[-1], 'color'),
                     markerfacecolor='none')

            tmp = cocopp.pprldistr.plotRLDistr(dictdim1[d], lambda fun_dim: targets(fun_dim)[j],
                              (targets.label(j)
                               if isinstance(targets,
                                             cocopp.pproc.RunlengthBasedTargetValues)
                               else targets.loglabel(j)),
                              marker=cocopp.genericsettings.line_styles[0]['marker'],
                              **cocopp.pprldistr.rldStyles[j % len(cocopp.pprldistr.rldStyles)])
            # modify the automatic legend: remover marker and change text
            plt.setp(tmp[-1], marker='',
                     label=targets.label(j)
                     if isinstance(targets,
                                   cocopp.pproc.RunlengthBasedTargetValues)
                     else targets.loglabel(j))
            # Mods are added after to prevent them from appearing in the legend
            plt.setp(tmp, markersize=15.,
                     markeredgewidth=plt.getp(tmp[-1], 'linewidth'),
                     markeredgecolor=plt.getp(tmp[-1], 'color'),
                     markerfacecolor='none')

        funcs = set(i.funcId for i in dictdim0[d]) | set(i.funcId for i in dictdim1[d])
        text = cocopp.ppfig.consecutiveNumbers(sorted(funcs), 'f')

        plt.axvline(max(i.mMaxEvals() / i.dim for i in dictdim0[d]),
                    marker='+', markersize=20., color='k',
                    markeredgewidth=plt.getp(tmp[-1], 'linewidth',))
        plt.axvline(max(i.mMaxEvals() / i.dim for i in dictdim1[d]),
                    marker='o', markersize=15., color='k', markerfacecolor='None',
                    markeredgewidth=plt.getp(tmp[-1], 'linewidth'))
        cocopp.toolsdivers.legend(loc='best')
        plt.text(0.5, 0.98, text, horizontalalignment="center",
                 verticalalignment="top", transform=plt.gca().transAxes)
        cocopp.pprldistr.beautifyRLD(evalfmax)
        cocopp.ppfig.save_figure(filename, dsList0[0].algId, subplots_adjust=dict(left=0.135, bottom=0.15, right=1, top=0.99))
        plt.close(fig)

cocopp.pprldistr.comp = customComp

def customMain(dsList, isStoringXMax=False, outputdir='', info='default'):
    """Generate figures of empirical cumulative distribution functions.

    This method has a feature which allows to keep the same boundaries
    for the x-axis, if ``isStoringXMax==True``. This makes sense when
    dealing with different functions or subsets of functions for one
    given dimension.

    CAVE: this is bug-prone, as some data depend on the maximum
    evaluations and the appearence therefore depends on the
    calling order.

    :param DataSetList dsList: list of DataSet instances to process.
    :param bool isStoringXMax: if set to True, the first call
                               :py:func:`beautifyFVD` sets the
                               globals :py:data:`fmax` and
                               :py:data:`maxEvals` and all subsequent
                               calls will use these values as rightmost
                               xlim in the generated figures.
    :param string outputdir: output directory (must exist)
    :param string info: string suffix for output file names.

    """
    testbed = cocopp.testbedsettings.current_testbed
    targets = testbed.pprldistr_target_values # convenience abbreviation

    for d, dictdim in sorted(dsList.dictByDim().items()):
        maxEvalsFactor = max(i.mMaxEvals() / d for i in dictdim)
        if isStoringXMax:
            evalfmax = cocopp.pprldistr.evalfmax
        else:
            evalfmax = None
        if not evalfmax:
            evalfmax = maxEvalsFactor
        if cocopp.pprldistr.runlen_xlimits_max is not None:
            evalfmax = cocopp.pprldistr.runlen_xlimits_max

        # first figure: Run Length Distribution
        filename = os.path.join(outputdir, 'pprldistr_%02dD_%s' % (d, info))
        fig = plt.figure()
        for j in range(len(targets)):
            cocopp.pprldistr.plotRLDistr(dictdim,
                        lambda fun_dim: targets(fun_dim)[j],
                        (targets.label(j)
                         if isinstance(targets,
                                       cocopp.pproc.RunlengthBasedTargetValues)
                         else targets.loglabel(j)),
                        evalfmax, # can be larger maxEvalsFactor with no effect
                        ** cocopp.pprldistr.rldStyles[j % len(cocopp.pprldistr.rldStyles)])

        funcs = list(i.funcId for i in dictdim)
        text = '{%s}, %d-D' % (cocopp.pprldistr.consecutiveNumbers(sorted(funcs), 'f'), d)

        plt.axvline(x=maxEvalsFactor, color='k') # vertical line at maxevals
        cocopp.toolsdivers.legend(loc='best')
        plt.text(0.5, 0.98, text, horizontalalignment="center",
                 verticalalignment="top",
                 transform=plt.gca().transAxes
                 # bbox=dict(ec='k', fill=False)
                )


        cocopp.pprldistr.beautifyRLD(evalfmax)
        cocopp.ppfig.save_figure(filename, dsList[0].algId, subplots_adjust=dict(left=0.135, bottom=0.15, right=1, top=0.99))
        plt.close(fig)

        # second figure: Function Value Distribution
        filename = os.path.join(outputdir, 'ppfvdistr_%02dD_%s' % (d, info))
        fig = plt.figure()
        cocopp.pprldistr.plotFVDistr(dictdim, np.inf, testbed.ppfvdistr_min_target, **cocopp.pprldistr.rldStyles[-1])
        # coloring right to left
        for j, max_eval_factor in enumerate(cocopp.genericsettings.single_runlength_factors):
            if max_eval_factor > maxEvalsFactor:
                break
            cocopp.pprldistr.plotFVDistr(dictdim, max_eval_factor, testbed.ppfvdistr_min_target,
                        **cocopp.pprldistr.rldUnsuccStyles[j % len(cocopp.pprldistr.rldUnsuccStyles)])

        plt.text(0.98, 0.02, text, horizontalalignment="right",
                 transform=plt.gca().transAxes) # bbox=dict(ec='k', fill=False),
        cocopp.pprldistr.beautifyFVD(isStoringXMax=isStoringXMax, ylabel=False)
        cocopp.ppfig.save_figure(filename, dsList[0].algId, subplots_adjust=dict(left=0.0, bottom=0.15, right=1, top=0.99))

        plt.close(fig)

cocopp.pprldistr.main = customMain
