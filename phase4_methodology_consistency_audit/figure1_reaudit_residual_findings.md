# Figure 1 Re-Audit — Residual Actionable Findings

**Scope:** Read-only re-check of `hossein_paper_revised.md` Sections 3.2--3.10 and `figures/proposed_architecture.svg`.

All 21 rendered arrows have source and destination assignments consistent with the stated high-level topology. The prior MSC-caption issue is resolved: Figure 1 now labels `B0` and `B` and delegates branch detail to Figure 2 (M161; S41--45). The `T -> T_hat` path and final `z0_half -> Upsample -> z0 -> sigmoid -> P` order are also now present (S72--94).

## Residual findings

### MODERATE — The Section 3.2 forward equation still passes `T`, not `T_hat`, to GDF

- **Locations:** M150--152; M287--299; M161; S72--81.
- **Evidence:** The overview equation specifies `z0 = PredictionHead(GDF(D1, T))` (M151). The detailed GDF definition instead first produces `T_hat` and uses `Cat(D1, T_hat)` for gate prediction and fusion (M287--299). Figure 1 likewise labels the input path as `T -> T_hat` and the GDF input as `Cat(D1, G odot T_hat)` (S72--81).
- **Action:** Make the overview path explicit: add `T_hat = DetailProjection(T)` and use `GDF(D1, T_hat)`. Alternatively, explicitly define `GDF(D1, T)` as including the projection, then alter the detailed text and figure labels to use that same interface. The first option best matches the current Figure 1 and Section 3.9.

### MINOR — Figure 1 still omits parts of the final tensor-size contracts

- **Locations:** M134--138; M161; S77--92.
- **Evidence:** The GDF box identifies `F_fuse` and `128 ch` but omits its `(H/2) x (W/2)` resolution (S77--79). `z0_half` lacks its `1 ch, H/2 x W/2` label (S83--85); `z0` lacks its `1 ch` label (S87--89). Table 2 defines all of these contracts (M134--138), while the caption says principal fusion points identify tensor dimensions (M161).
- **Action:** Add `F_fuse: 128 ch, H/2 x W/2`, `z0_half: 1 ch, H/2 x W/2`, and `z0: 1 ch, H x W`. Optionally label `P: 1 x H x W` for a fully explicit output contract.

### MODERATE — Auxiliary heads have correct sources but no depicted full-resolution logit outputs

- **Locations:** M317--325; S96--107.
- **Evidence:** Purple dashed arrows correctly attach auxiliary heads to `D2`, `D3`, and `D4` (S105--107). The boxes do not show the required `Conv1x1 -> Upsample(H,W) -> z2/z3/z4` paths in M320--322.
- **Action:** Add `z2`, `z3`, and `z4` output labels with `1 ch, H x W`, or place `1x1 conv + Upsample(H,W)` inside each auxiliary-head box.

## Scoped editorial decision and revision roadmap

**Decision: Minor Revision.** The corrected main topology is consistent. Resolve the one `T`/`T_hat` notation inconsistency and complete the stated tensor and auxiliary-output labels before treating Figure 1 as a full architecture specification.

1. Correct the Section 3.2 GDF argument to `T_hat` (or define the alternate interface consistently).
2. Complete `F_fuse`, `z0_half`, and `z0` tensor dimensions in Figure 1.
3. Label the three upsampled auxiliary logits.

## AI disclosure

This re-audit was generated with AI assistance from read-only comparison of the specified Markdown manuscript and SVG source. It did not inspect implementation code or execute the model.
