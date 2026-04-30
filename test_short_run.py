#!/usr/bin/env python
"""Quick 5-iteration test run of the optimizer."""

import sys
sys.path.insert(0, '.')
import optimizer
import numpy as np

print('\n' + '='*60)
print('SHORT OPTIMIZATION TEST (5 calls)')
print('='*60 + '\n')

# Initialize configuration
optimizer.initialize_optimization_config(n_inner_layers=7, optimize_disks=False)

# Get baseline parameters
thetas_r = np.array([optimizer.BASE_LAYERS[i]["rPos"] for i in optimizer.FREE_BARREL_IDX])
thetas_l = np.array([0.5*(optimizer.BASE_LAYERS[i]["xMax"]-optimizer.BASE_LAYERS[i]["xMin"]) for i in optimizer.FREE_BARREL_IDX])
thetas = np.concatenate([thetas_r, thetas_l])

# Write baseline and compute initial loss
optimizer.write_geo_from_theta(thetas)
optimizer.run_root_metrics(optimizer.GEO_OPT, optimizer.METRICS_FILE)
L0 = optimizer.score(thetas)
print(f'\n*** Baseline loss: {L0:.6f} ***\n')

# Run short optimization: 5 total calls, 3 random initialization
print('Starting short optimization run...\n')
result = optimizer.run_bayes_optimization(n_calls=5, n_initial_points=3)

print('\n' + '='*60)
print('TEST COMPLETE!')
print('='*60)
print(f'Initial loss: {L0:.6f}')
print(f'Final loss:   {result.fun:.6f}')
print(f'Improvement:  {(L0 - result.fun)/L0*100:.2f}%')
