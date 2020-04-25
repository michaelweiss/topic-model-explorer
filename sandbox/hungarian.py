import numpy as np
from scipy.optimize import linear_sum_assignment

# example KL-divergence distance matrix between topics
cost = np.array([[0.86885,	1,	0.964983,	0.774382,	0.620403],
[0.812494,	0.831729,	0.717108,	0.720588,	0.846783],
[0.757327,	0.808635,	0.954579,	0.847733,	0.795859],
[0.771076,	0.691765,	0.934884,	0.865098,	0.909909],
[0.82239,	0.759648,	0.717514,	0.890027,	0.996661]])

row_ind, col_ind = linear_sum_assignment(cost)

print(col_ind)
print(cost[row_ind, col_ind].sum())