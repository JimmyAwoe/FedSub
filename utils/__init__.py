from .linear import SubScafLinear
from .subscafsgd import SubScafSGD
from .random_matrix_gene import *
#from .subscafadam import SubScafAdam
from .common import log, init_process_group, set_seed
from .replace_modules import replace_with_subscaf_linear, outer_update