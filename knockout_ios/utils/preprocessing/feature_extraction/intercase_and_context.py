import os

from copy import deepcopy

import pandas as pd

from .times_generator import TimesGenerator

def get_default_params():
    # Times allocator parameters
    parms = dict()
    parms['imp'] = 1
    parms['max_eval'] = 12
    parms['batch_size'] = 32 # Usually 32/64/128/256
    parms['epochs'] = 200
    parms['n_size'] = [5, 10, 15]
    parms['l_size'] = [50, 100] 
    parms['lstm_act'] = ['selu', 'tanh']
    parms['dense_act'] = ['linear']
    parms['optim'] = ['Nadam']
    parms['opt_method'] = 'rand_hpc' # bayesian, rand_hpc
    parms['reschedule'] = False # reschedule according resource pool ocupation
    parms['rp_similarity'] = 0.80 # Train models

    # modelo ‘dual_inter’ habilita el calculo los features al principio y al final de la ejecución de la actividad 
    # (cualquier otro valor solo calcula al principio). Hay otro parámetro llamado all_r_pool si es True calcula un feature 
    # por cada pool de recurso en False solo incluye el valor del pool actual

    parms['model_type'] = 'dual_inter' # basic, inter, dual_inter, inter_nt
    parms['all_r_pool'] = False # only intercase features
    
    return parms

def extract(log, _model_type='dual_inter', _all_r_pool=False):
    
    parms = get_default_params()
    parms['model_type'] = _model_type # basic, inter, dual_inter, inter_nt
    parms['all_r_pool'] = _all_r_pool # only intercase features

    generator = TimesGenerator(deepcopy(log.data), parms)

    with_intercase, res_analyzer = generator._add_intercases()
    with_contextual = generator._add_calculated_times(with_intercase)

    return with_contextual, res_analyzer


def extract_only_contextual(log, _model_type='dual_inter', _all_r_pool=False):
    parms = get_default_params()
    parms['model_type'] = _model_type  # basic, inter, dual_inter, inter_nt
    parms['all_r_pool'] = _all_r_pool  # only intercase features

    generator = TimesGenerator(deepcopy(log.data), parms)

    with_contextual = generator._add_calculated_times(pd.DataFrame(log.data))

    return with_contextual