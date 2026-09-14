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

## The two open bugs (both real, both repro'd)
1. Generation deadlock (run17, step-0 gen): jitted generate() with
   jax.random.categorical deadlocks inside TPU driver - one driver thread
   pins 100% CPU for 20+ min, no I/O, last compiled op jit__argmax. Not
   diagnosed further; py-spy blocked (no ptrace in container).
2. val_loss OOM at 16k vocab (dbg3): jit_ev needed 126MB contiguous with
   119MB free; val batch was 64x512x16384 fp32 logits. FIXED in local file:
   val batch 64 -> 16 (train500m_bpe.py line 158). Untested - dbg4 was the
   test of this fix when the kernel died.

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
3. Generation at step 0 fires first - that is the deadlock test. If it hangs
   again (no GEN lines in ~3min), the fix to try: batch the generate loop with
   lax.scan, or pre-compile generate_step with static shape (1,SEQ) + padded
   KV, or fall back to NAVI_GEN=0 + offline generation from ckpt.
4. Watch: first CKPT line at step 499 (~17min after stepping starts).
5. Disk: need ~9GB free for keep-2 ckpts (4.6GB x2) + 4GB hf_cache; delete
   genpolish_params.pkl/tokenizer_corpus.txt if present.

## Monitoring lesson
Printed tok/s is cumulative-average (diluted by 8min compile): judge speed by
step-line spacing (~2.1s) not the printed number.