import sys
sys.path.insert(0, "/kaggle/working")
sys.path.insert(0, "/kaggle/working/experiments")
from fineweb_data import FineWebFeed
import numpy as np
import time
import os

feed = FineWebFeed(n_val_docs=200)
rng = np.random.default_rng(0)
for _ in range(150):
    try:
        w = feed.batch(rng, batch=8, seq=512)
        break
    except RuntimeError:
        time.sleep(2)
t0 = time.time()
n = 0
for k in range(600):
    w = feed.batch(rng, batch=256, seq=512)
    n += w.size
dt = time.time() - t0
print(f"sustained: {n/dt/1024/1024:.2f} MB/s consumed over {dt:.0f}s; buffer={len(feed.train_buf)//1024//1024}MB err={feed.err}")
os._exit(0)