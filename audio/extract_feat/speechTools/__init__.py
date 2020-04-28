"""Package initialisation module.

This initialisation module mainly contains sub modules imports for direct access to sub modules fonctions.
Therefor, folowing calls are aloud : speechTools.subModule.function() or speechTools.function().

"""

import numpy as np
import scipy as sp

from speechTools.clustering import *
from speechTools.detect_peaks import *
from speechTools.energy import *
from speechTools.features import *
from speechTools.io import *
from speechTools.pitch import *
from speechTools.signal import *
from speechTools.speech import *
from speechTools.stats import *
from speechTools.trans import *
from speechTools.yin import *

