#!/usr/bin/env python
"""Diagnose nozzle clearance violations."""

import sys
sys.path.insert(0, '.')
import optimizer
import numpy as np

print('='*60)
print('NOZZLE CLEARANCE DIAGNOSTIC')
print('='*60)

# Initialize configuration
optimizer.initialize_optimization_config(n_stations=4, optimize_disks=False)  # the 4 VTX doublets

# Get baseline parameters
thetas = optimizer.baseline_theta()
thetas_r = thetas[:optimizer.N_FREE_BARREL]
thetas_l = thetas[optimizer.N_FREE_BARREL:]

layers = optimizer.build_layers_from_theta(thetas)

print('\nChecking barrel layers against nozzle profile...\n')
print(f"{'Layer':<6} {'R (m)':<8} {'z_end (m)':<10} {'r_noz (m)':<10} {'Violation (mm)':<15}")
print('-'*60)

total_violation = 0
for i, L in enumerate(layers):
    if L["tyLay"] != 1:  # Only barrel layers
        continue

    R_b = L["rPos"]
    z_end = max(abs(L["xMin"]), abs(L["xMax"]))  # half-length
    r_noz = float(optimizer.nozzle_profile(z_end))

    violation_m = r_noz - R_b
    violation_mm = violation_m * 1000  # convert to mm

    status = "❌ VIOLATES" if violation_m > 0 else "✓ OK"

    print(f"{L['label']:<6} {R_b:<8.4f} {z_end:<10.4f} {r_noz:<10.4f} {violation_mm:>10.2f} mm  {status}")

    if violation_m > 0:
        total_violation += violation_mm

print('-'*60)
print(f'\nTotal clearance violations: {total_violation:.2f} mm')

print('\n' + '='*60)
print('NOZZLE PROFILE FUNCTION')
print('='*60)
print('r_noz(z) = 0.1763 * z  for z < 1.0 m')
print('r_noz(z) = 0.08474 * z + 0.09156 m  for z >= 1.0 m')
print('\nAt z=0.065 m (baseline half-length):')
print(f'  r_noz = {optimizer.nozzle_profile(0.065):.4f} m = {optimizer.nozzle_profile(0.065)*1000:.2f} mm')
print('\nFor a barrel at R=0.029 m to clear the nozzle:')
print(f'  Maximum z_end = R/0.1763 = {0.029/0.1763:.4f} m = {0.029/0.1763*1000:.2f} mm')
