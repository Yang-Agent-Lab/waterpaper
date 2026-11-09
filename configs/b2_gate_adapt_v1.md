# RESEARCH EXECUTION SPEC: B2共享与状态调节

Status: APPROVED. Revision 1.0. Change ID: WP-20261018-B2-GATE-ADAPT-V1.
Authority: 用户2026-10-18明确批准先准备缓存、模块健康与作用检查，验证后在预算内运行B/C；A复用，不重跑旧实验。

## Question and interpretation

在固定已有历史编码、分段、方差候选及均值/去噪器参数的条件下，调整状态修正的强度与衰减速度是否有价值？按起报状态调整是否优于所有起报共用的调整？
B对A是额外门控拟合的开发收益；C对B为直接机制对照。单种子开发select探索，不能称稳定优势、新模块原创或公平外部领先。负结果是完整结果，不追加预算救正。

## Starting point and isolation

- Local execution checkout: . ; reuse its clean branch codex/b2-temporal-state-v1 because previous job is complete.
- Baseline source: 3bdec5d1c3926d8d211cfff4112c63b2194dbdb3. Never change running source or old artifacts.
- Remote execution: . . All new research files/results/cache stay under ..
- A selected fixed_segments checkpoint: ./results/experiments/b2_temporal_state_v1_20261018/formal/formal_3bdec5d_20261018/artifacts/fixed_segments/best.pt
- A SHA256: 155043d2b56e108d2b13c8870be57db5dadc1fd6d3c91d219d45526f6a9b4e41 ; selected update 2000, formal training originally 3000. Do not train A.
- A 128-draw selection score: 0.02122260879547767, original selection seed600002. Reuse as candidate update0; exact same rows/noise protocol required.
- A final256-draw primary Brier: 0.021088611769596522. Reuse saved predictions/select_rows/canonical labels, final seed900002; no repeat A full sampling.
- Original C checkpoint, author DiffPTS source, environment and bank remain as configs/b2_temporal_state_formal_v1.json.
- New output root: ./results/experiments/b2_gate_adapt_v1_20261018 . Unique named run dirs under execution worktree for run_versioned, root may contain cache/budget references.

## Structures and trainable scope

Let mu0 and g0 be the original C mean and base variance; gs is A's state variance before its fixed lead mixture. All original C and A state-branch parameters remain fixed/eval. The original denoising native loss and sampler remain unchanged, including differentiability through variance to the new gate.

w_h=clip(a*exp(-(h-1)/tau),0,1); g_h=(1-w_h)*g0_h+w_h*gs_h. Use stable arithmetic, original variance floor1e-8, original physical projection, future1..24h.

- A: a=1,tau=6h, reuse only.
- B shared: two trainable scalar logits u_a,u_tau. a=2*sigmoid(u_a), tau=1+23*sigmoid(u_tau). Initial u_a=0,u_tau=log(5/18) so a1/tau6.
- C state: z is the mean across24 historical hourly tokens of A's existing width14 temporal encoder (before adding segment tokens); signed-log P/Q and position exactly as A. Normalize the14 z features by fit-only means/population standard deviations (floor1e-6), fitted once on unique fit rows used by the approved1000-update exposure. Two linear14->1 readouts supply u_a,u_tau. Total30 trainable parameters. Weight0/biases as B ensure initial reproduction. No extra encoder/hidden layer/lead features/labels in gate input.
- Clip is the declared architecture, not an engineering error; log gate saturation rates and gradients. Frozen parameters receive no optimizer updates; absence of their gradient is expected, not a failure.
- Same bank, Q targets, likelihood + denoising loss, no Brier supervision or new auxiliary objective. Do not detach/no_grad the denoiser's dependence on new variance even though its weights are fixed.

## Data, cache and pairing

Bank ./results/experiments/pointwise_full_pack_108f9a0/artifacts/pq ; manifest SHA2957aabab9a36247168e3ee05a6cd0f4490ac2f31395b2cd53c384668d7ce81c.
Same original C accepted_bank_indices exposure; first1000 optimizer updates x64, same global fit indices and same noise seeds for B/C. Keep original bank fit arrays and source mapping intact. Complete original-C China select view1706 rows/9stations/17 source-event units, row order SHA5aac1a4e22cb38b233ba6381cc747757e03d68a1b0a8256139431cba52660707.
No held-out/test/calibration/old huge-peak access, no future real P, no new downloads/data/split/country changes. A and old five groups preserved read-only.

Cache only unique fit rows touched by those1000 steps plus complete China select. Store role+sample_id+source global index, x/y or read-only bank references, mu0,g0,gs,z raw+normalized, canonical select metadata, normalization fit identities and statistics. Store float32 frozen model outputs, immutable weight/source/bank/scale/row identities, dtype and checksums. Do NOT invert A mixed variance to recover gs; recompute frozen pre-gate gs once. Do not cache new-gate-dependent denoiser outputs or pretend old draws are new predictions.
Budget starts before first new server computation (read-only metadata excluded), accumulate CPU/model contract checks, cache construction, smoke, formal, evaluation and numerical diagnostics under one1200-second receipt; GPU lock uses existing pointwise_gpu0.lock. Editing/transfers/human wait do not consume compute, but active compute does. Never reset on retry. Cache stays server, no local model/data copies.

## Prechecks: phase1, then return to parent

Commit before any new server code run; use src/run_versioned.py. Existing environment ./environments/pointwise_benchmark_v1/bin/python. Default ssh teacherbox/scp; if it fails, inspect accurate state before retry and use only already verified existing alternate alias, never change global SSH or restore wakes.

Required before formal:
1. Verify A/C checkpoint and bank identities, exact fit exposure/select pairing. Check no prior active/completed same-ID stage before start; preserve failures.
2. B2/C30 trainable parameters; all A parameters exact unchanged; initial conditions reproduce A on finite fit/select probes with recorded float32 tolerances (suggest atol1e-6 rtol1e-5), fixed-noise representative draws likewise. Do not demand bitwise floating-point equality.
3. Cache-vs-live mu0/g0/gs/z and initial native loss/selected representative predictions agree on the same16 fixed fit and16 select samples; ensure fit-only normalization, finite floor and no source/role mismatch. Cache hit/miss counters and elapsed phase time saved.
4. Trace finite gradients from both native loss components through frozen denoiser to new gate on a representative batch (no optimizer update); report zero/unused dimensions honestly.
5. Save module health for temporal encoder, segment pool, lead query/readout, original/state variance output, new amplitude and tau heads, frozen mean and denoiser. Each includes applicable output finite fraction, variance/range, gradient and actual update, expected frozen status; parameterless stages marked applicable output only. Fixed weights do not require gradients. Log a/tau/w distributions and clipping fraction by lead, and mean/variance fingerprints.
6. Fixed-weight action probes: at most24 predeclared select origins covering ordinary/onset/sustained stages with station/event identities, all24 leads,128 paired draws. Use saved A predictions for restored-A comparison, preserve128/256 draw counts and avoid presenting unmatched-draw differences as exact intervention effects. Phase1 small representative cached/live replay may generate initial128 draws for these24 origins only. Return-to-A conditions equality is the exact action contract; saved final A256 is reference, not a paired128 difference. Any new matched alternative shares128 RNG draws. Keep probe cost within budget.
7. Smoke B/C two optimizer updates each, total4 attempted max (failures count); fresh start for formal. Record parameters, cache hits, loss/gradient/update and finite condition changes, frozen fingerprints, saving+sampling+scoring/report path end-to-end on limited fixed probes, not full China repeated smoke.

Phase1 returns commit/config/cache/budget/precheck receipts and must STOP before formal until parent verifies existing evidence. Parent confirmation is an internal evidence check under user's authorization, not another user permission request.

## Formal budget and selection: phase2 after parent check

- B/C seed2, each1000 attempted updates max; total2000 plus smoke4 separate. Batch64, Adam1e-3, clip10. No optimizer updates on A, no LR/tau search, no extra seeds/trials/epochs.
- Candidate updates0 (reuse original A128 score),500,1000. Complete China select128 draws per nonzero candidate, original random pairing seed600002/eval batch32/row order. Lowest primary score, earliest tie; update0 candidate shared by both. Frozen cached/live equivalence must support its reuse.
- Final chosen model complete China select256 draws, seed900002. If chosen0 use A saved original256 NPZ directly. Save raw and projected draws, probability/truth/observed/physical+normalized thresholds, conditions, sample IDs, scores, components, selection, last/best state and hashes. Never rescale A draws as changed-g output.
- Keep native loss and select loss definitions identical, log both native components. Health update records at0/250/500/1000 on bounded representative batches; batch gradient nonzero fraction is a measured diagnostic, no invented overall health score.
- At selected C, fixed-weight state-use probe on the same<=24 origins: replace standardized gate z by fit mean0, keeping A encoder/variance candidate fixed;128 paired draws for factual vs ablated. An all-gate-to-A intervention uses exact A conditions and saved A256 reference, no full A resampling. Scalar feature/condition output probes may inspect individual stages without pretending to be retrained contribution.
- Require report_bundle schema health+interventions actual rows, no empty sections silently labeled complete. Reuse training_report_v1 learning/prediction/risk pages; module diagnostics retained numerically, not mandatory chart. TrainingRecorder integration reuses existing fixed recorder and sender; no new wake service. Existing A learning curve reused with history label; no invented fresh A curve.
- Single1200s compute limit across phases with attempt counts/persistent receipt, including failures. Save checkpoint+partial if cap reached; do not silently restart. No inference/health probe expands held-out/data/group budget.

## Primary results and decision

Full24h Q95/Q99 Brier, China station->sourceevent->origin equal weights and mean leads/thresholds, lower better. Paired B-A,C-A,C-B absolute and relative effects, exact roles, original DiffPTS C and old five variants read-only reference. Current A reference0.021088611769596522, original C0.021693367399576936. No current equal-total-budget external model; external improvement field— with reason.
Also report1-3/4-6/7-12/13-24h, Q95/Q99 onset/middle/positive/negative with actual support and adverse values. Health is connectivity/update behavior; fixed-weight probes dependency; trained B/C comparison is incremental development performance. No arbitrary pass threshold, no selecting favorable phase after seeing results.
Prefer B if C offers no meaningful coherent added value; retain C only as exploratory if overall gain and phase/ordinary tradeoffs justify its added dependence. If neither beats A or only causes costly tradeoffs, retain A and state what weakens. Negative/partial are eligible outcomes; no additional training.

## Independent acceptance and delivery

Parent did not implement engineering; parent independently checks immutable implementation and cache equivalence before formal, then reconstructs primary hourly probabilities/Brier and phase scores from saved raw draws/canonical labels; verifies chosen0 reuse or500/1000 selection, immutable A fingerprints, budget and actual health/intervention payloads. Scope review is required for negative as well as positive results. It may run CPU-only audit in a distinct parent audit checkout, with no new model inference/training.
Engineer owns new b2_gate_* files, focused checks/configs and allowed new run output; never rewrites old five-group files/results. Main owns current-state docs, Science/task records and final independent audit. No concurrent same-file writers, no subdelegation.
Commit/source/config/input/cache/run identities, actual timing, attempts, logs, negative cases, source-code diff and final results must be traceable. First return phase1 short receipt with real files, no long logs; formal receipt after all2 end. Preserve all partial and failed evidence.
No pushing/merging/publishing, data deletion/migration, training outside this plan, new user-side chat, old wake/cron/heartbeat restart, or global SSH changes.
