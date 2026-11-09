# Public Waterpaper development rules

- Read README.md, CONTRIBUTING.md, and the relevant module/configuration before editing.
- Preserve existing work and source provenance. Scope each change to a clear research or engineering question.
- Keep hourly flood state risk, any-exceedance window risk, and first-event-time risk distinct.
- Do not silently change labels, event splits, data roles, normalization, sampling, model selection, or scoring definitions.
- Keep data, weights, predictions, credentials, and private collaboration records out of Git. Check the actual staged file list before publishing.
- Prepare experiments with the maintainer; do not launch full training, use held-out test data, upload records, or contact data providers just because a historical script exists.
- Write run artifacts directly to results/experiments/<unique-run-id>/; use src/run_versioned.py for versioned formal execution from a clean committed tree.
- Run scientific computation and plotting in the agreed compute environment. Do not claim results from static checks or synthetic interface checks.
- Document observed behavior, failed/partial/not-evaluable states, test scope, and remaining dependencies precisely.
- Historical configurations need explicit path/data/weight review. Never bypass fingerprint checks to make a stale run start.
