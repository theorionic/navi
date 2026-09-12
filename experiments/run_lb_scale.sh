#!/bin/bash
# lb scale check: two arms, sequential (TPU v3-8, one process at a time).
# arm 1: 65k-fact anchor (expect ~100% fresh - validates lb doesn't hurt anchor)
# arm 2: 16.8M-fact scale (the VERDICT failure; success = fresh >> 0.4% chance)
set -e
cd /kaggle/working/code
echo "=== ARM 1: nonce=64 ==="
NAVI_NONCE=64 NAVI_STEPS=4000 NAVI_LB=0.01 python3 -u experiments/lb_scale.py 2>&1 | tee /kaggle/working/lb_scale_64.log
echo "=== ARM 2: nonce=1024 ==="
NAVI_NONCE=1024 NAVI_STEPS=4000 NAVI_LB=0.01 python3 -u experiments/lb_scale.py 2>&1 | tee /kaggle/working/lb_scale_1024.log
echo "=== BOTH ARMS DONE ==="