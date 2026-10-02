#!/usr/bin/env python
"""Test script to compute baseline loss without running full optimization."""

import sys
sys.path.insert(0, '.')
import optimizer
import numpy as np

print('='*60)
print('BASELINE LOSS TEST')
print('='*60)

# Initialize configuration
optimizer.initialize_optimization_config(n_stations=4, optimize_disks=False)  # the 4 VTX doublets

# Get baseline parameters
thetas = optimizer.baseline_theta()
thetas_r = thetas[:optimizer.N_FREE_BARREL]
thetas_l = thetas[optimizer.N_FREE_BARREL:]

print(f'\nBaseline parameters:')
print(f'  Radii (m):       {thetas_r}')
print(f'  Length fractions: {thetas_l}')

# Write baseline geometry
optimizer.write_geo_from_theta(thetas)
print(f'\n✓ Wrote geometry to: {optimizer.GEO_OPT}')

# Try to run ROOT metrics
print(f'\nRunning ROOT to compute metrics...')
try:
    optimizer.run_root_metrics(optimizer.GEO_OPT, optimizer.METRICS_FILE)
    print(f'✓ ROOT metrics computed successfully')

    # Compute loss
    print(f'\nComputing loss from metrics...')
    loss = optimizer.compute_loss_from_metrics(optimizer.METRICS_FILE)
    print(f'\n{"="*60}')
    print(f'BASELINE LOSS: {loss:.6f}')
    print(f'{"="*60}')

    # Also compute full score with penalties
    print(f'\nComputing full score (with nozzle penalty)...')
    full_score = optimizer.score(thetas)
    print(f'\n{"="*60}')
    print(f'FULL BASELINE SCORE: {full_score:.6f}')
    print(f'{"="*60}')

except Exception as e:
    print(f'\n✗ ERROR: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

print('\n✓ Baseline test completed successfully!')
