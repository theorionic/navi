# bpe500m run state @ kernel loss #2 (2026-09-14 ~20:10 IST)

## What was running
train500m_bpe.py, 20k-step target, gen@1000, NAVI_GEN=1 attempt (dbg4, pid 106888)
resumed from ckpt_bpe500m_step000999.pkl. Died with the kernel (relay agent
"agent is not connected" from 20:04 IST; four probes over 10 min all dead).
Second kernel loss of the day; first was ~13:41 (run18/19 era).

## Verified state before the drop
- Training proven: step 1300+ reached, loss 4.53/bpc 6.54 at 1.3k steps,
  ~2.1s/step instantaneous (≈60k tok/s), buffer full 1024MB.
- Checkpoint path proven: ckpt_bpe500m_step000{499,999}.pkl both written and
  pickle-loadable (4.6GB each, keep-2 rotation).
- VAL evals ran: VAL@0 13.22, VAL@1000 7.07 bpc (held-out).
- Data resume exact (PhaseFeed cursor skip verified twice).

## Bug status after run 1
1. Generation livelock - FIXED (commit 36b5403). Rewritten as a single
   jitted lax.scan (_gen_scan in train500m_bpe.py): 4 prompts batched,
   sampled on-device, per-step seed=step_i+42, EOS masked, ONE device_get
   per burst. CPU smoke-tested end-to-end incl. the real tokenizer surface.
   Root cause: the old eager loop did ~2.5k dispatch/sync round-trips per
   burst -> driver-thread livelock on a never-completing device->host
   transfer. Not a compile issue (all compiles finished).
2. val_loss OOM at 16k vocab - FIXED same commit (val batch 64->16).
3. pkm._hash_path NameError ('out' undefined) - FIXED same commit; found
   by the smoke test. Hash-path forward regression-tested.
   Remaining TPU-only risk: scan compile is ~4min extra at first gen; if
   the hang somehow recurs, fall back to NAVI_GEN=0 + offline generation.

## Local files (authoritative, checksummed)
- experiments/train500m_bpe.py  6f5d0388  (val-batch fix in, gen re-enabled via NAVI_GEN=1 default path)
- experiments/grain_parquet_data.py  486def05
- experiments/tokenizer_16k.json  (16k BPE, kernel-verified)

## Relaunch recipe (next kernel)
1. relayfs_write_file all three files to kernel
   (/kaggle/working/code/experiments/{train500m_bpe.py,grain_parquet_data.py->tok/},
    /kaggle/working/tokenizer_16k.json)
2. NAVI_DATA_STATE=/kaggle/working/data_state_fresh.json \
   NAVI_HF_CACHE=/kaggle/working/hf_cache \
   NAVI_STEPS=20000 NAVI_GEN=1 NAVI_GEN_EVERY=1000 \
   nohup python3 code/experiments/train500m_bpe.py > train.log 2>&1 &
   (fresh start - no ckpts survive; ~8min compile, then ~2.1s/step, 20k steps ~ 12-17h)
3. Generation at step 0 fires first - with the new scan path, expect a
   ~4min compile after VAL@0, then 4 GEN lines within seconds of each
   other. If no GEN line ~10min after VAL@0: kill, relaunch with
   NAVI_GEN=0 (training unaffected), generate offline from ckpt later.
4. Watch: first CKPT line at step 499 (~17min after stepping starts).
5. Disk: need ~9GB free for keep-2 ckpts (4.6GB x2) + 4GB hf_cache; delete
   genpolish_params.pkl/tokenizer_corpus.txt if present.

## Monitoring lesson
Printed tok/s is cumulative-average (diluted by 8min compile): judge speed by
step-line spacing (~2.1s) not the printed number.