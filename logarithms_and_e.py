"""
log => ln(x) = log_e(x) # base is e not 10 (important)
log => solving for x => e ** x = b
    here, b -> input, x -> output
"""

import numpy as np

b = 5.2

print(np.log(b))  # ln(5.2) = 1.6486586255873816
# meaning e ** 1.6486586255873816 = 5.2
