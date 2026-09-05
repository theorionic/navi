import sys
sys.path.insert(0, "/kaggle/working")
sys.path.insert(0, "/kaggle/working/experiments")
from fineweb_data import FineWebFeed
import numpy as np
import time
import os

# 600 x (256x512) = 78MB per second would need the buffer to GROW at 78MB/s;
# this test checks network fill rate vs consumption rate over a longer window.
feed = FineWebFeed(n_val_docs=200)
rng = np.random.default_rng(0)
for _ in range(150):
    try:
        w = feed.batch(rng, batch=8, seq=512)
        break
    except RuntimeError:
        time.sleep(2)
marks = []
t0 = time.time()
n = 0
k = 0
while time.time() - t0 < 60:
    w = feed.batch(rng, batch=256, seq=512)
    n += w.size
    k += 1
    if k % 100 == 0:
        with feed._lock:
            marks.append((time.time() - t0, len(feed.train_buf) // 1024 // 1024))
dt = time.time() - t0
with feed._lock:
    final_buf = len(feed.train_buf) // 1024 // 1024
print(f"consumed {n/1024/1024:.0f}MB in {dt:.0f}s = {n/dt/1024/1024:.1f} MB/s; buffer trajectory {marks} -> {final_buf}MB; docs_err={feed.err}")
os._exit(0)