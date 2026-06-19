# Issue #12 recursive co-activation MACE artifact bundle

This bundle preserves the Issue #12 BFCL recursive co-activation MACE run state after v13 canonical completion.

- GitHub issue: https://github.com/tokenbender/prism-capability-extraction/issues/12
- Checkpoint comment: https://github.com/tokenbender/prism-capability-extraction/issues/12#issuecomment-4702835490
- Source commit: `07b46cdd41648dd83fbe8180752085bd59adafb5` on `issue12-recursive-coactivation-mace`
- Full anchor: `664/1007`
- MACE-90 threshold: `598/1007`
- Best v13 score: `607/1007` at `k=141250` (`91.42%` recovery)
- Smallest confirmed v13 MACE-90: `600/1007` at `k=140875` (`90.36%` recovery, `31.8457%` of MLP channels)
- Label: compressed/non-uniform MACE-90 incumbent, not clean MACE-90. Category floors fail `java`, `javascript`, and `live_simple`.
- Terminology: SFT-adapter-conditioned mask search using prior #6 `b007` adapter; no additional #12 SFT/training.

Key paths inside this bundle:

- `runs/issue12_recursive_coactivation_mace/`
- `data/bfcl_single_call/`
- `code/scripts/`
- `docs/`
- `MANIFEST.json`
- `SHA256SUMS`
