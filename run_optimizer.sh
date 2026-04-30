#!/bin/bash
# Wrapper script to run optimizer with crilin environment

PYTHON=/lustre/cmswork/fnardi/anaconda3/envs/crilin/bin/python
ROOT=/lustre/cmswork/fnardi/anaconda3/envs/crilin/bin/root

# Update PATH to include ROOT
export PATH=/lustre/cmswork/fnardi/anaconda3/envs/crilin/bin:$PATH

echo "Running optimizer with crilin environment"
echo "Python: $PYTHON"
echo "ROOT: $ROOT"
echo ""

exec $PYTHON optimizer.py "$@"
