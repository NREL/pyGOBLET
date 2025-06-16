import os
import warnings
from collections import OrderedDict

import numpy as np
import cocopp
import cocopp.pproc
import cocopp.testbedsettings

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
            found = False
            for group_name, ids in cocopp.testbedsettings.current_testbed.func_cons_groups.items():
                if i.funcId in ids:
                    res.setdefault(group_name, cocopp.pproc.DataSetList()).append(i)
                    found = True
            if not found:
                warnings.warn('Unknown function id: %s' % i.funcId)
    return res

cocopp.pproc.DataSetList.dictByFuncGroupSingleObjective = customDictByFuncGroupSingleObjective

# Custom html file for updated titles, labels, and descriptions
custom_html_path = os.path.abspath("custom_titles.html")
cocopp.genericsettings.latex_commands_for_html = os.path.splitext(custom_html_path)[0]
