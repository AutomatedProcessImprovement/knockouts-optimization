from .Simod.src.simod import structure_miner

# TODO: clean up this dependency on 2 versions of Simod...
import variability_analysis.preprocessing.external.feature_extraction as intercase_and_context
from variability_analysis.preprocessing.external import config_data_from_file, Configuration, ReadOptions, LogReader
from variability_analysis.preprocessing.external.Simod.src.simod.configuration import config_data_with_datastructures
