# Arrow-by-Arrow Methodology Consistency Audit

**Scope.** Read-only comparison of `hossein_paper_revised.md` Sections 3.2--3.10 and Figure 1 caption with `figures/proposed_architecture.svg`. Manuscript and SVG source were not modified.

**Location convention.** `M` = manuscript line; `S` = SVG line.

## Audit conclusion

**Scoped editorial decision: Major Revision.** The main encoder-to-decoder topology, all three skip-source assignments, the Detail Branch input, and all three auxiliary-head attachments are directionally consistent. However, Figure 1 does not expose several tensor transitions that the methods define, and its caption incorrectly says that the four MSC branches are shown. In particular, the diagram cannot verify the required `T -> T_hat -> GDF` projection or the required `z0_half -> Upsample(H,W) -> z0 -> sigmoid -> P` path. These gaps prevent Figure 1 from serving as a self-consistent and reproducible architecture specification.

No reversed main-path arrow, swapped skip level, or auxiliary-head source mismatch was found.

## Component and connection checklist

| Expected component or contract | Present in SVG? | Connected consistently? | Evidence and finding |
|---|---|---|---|
| Normalized input `X`, `3 x H x W` | Yes, at high level | Yes | `Input X` is labeled `3 x H x W` (S19--21), matching `X` in M105--113. Normalization itself is not drawn; this is acceptable only because the box is explicitly named `X`, which the manuscript defines as normalized. |
| Four ConvNeXt encoder outputs `F1`--`F4` | Yes | Yes | Stage labels, widths, and resolutions agree with Table 2 (S23--39; M123--129). |
| Residual bottleneck `F4 -> B0` | Partial | Not independently verifiable | A single combined box is reached from `F4` (S41--44), but neither `B0` nor the residual operation is shown. M143--145 and M183--191 define a distinct `B0` transition. |
| MSC `B0 -> B`, including four branches | Partial | Not independently verifiable | The SVG names only a combined `Residual bottleneck + MSC` box (S41--44). It has no `B0`, `B`, `R`, `C1`, `C3`, `C5`, or `Cg` nodes/branches, although M195--218 defines them and the Figure 1 caption says the four branches are shown (M161). |
| Progressive decoder `B -> D4 -> D3 -> D2 -> D1` | Yes, at block level | Yes, at block level | The solid chain is present (S58--61); output resolutions and `D4`--`D2` channel widths agree with M130--134 and M263--269. The individual `U4`, `U3`, and `U2` transitions are not drawn. |
| Three LRS-E applications | Yes, as composite blocks | Partly | `Decoder 4/3/2 + LRS-E` blocks are present (S46--54). The figure does not display the required internal order `Decoder_l -> U_l -> Cat(U_l,F_l) -> LRSE_l` from M146--148 and M263--268. It is therefore a valid high-level summary, but not an arrow-level representation of those equations. |
| `F3 -> LRS-E4`, `F2 -> LRS-E3`, `F1 -> LRS-E2` skips | Yes | Yes | The gray paths and labels map each skip to the correct level (S63--68), matching M146--148 and M263--268. |
| No encoder skip into `D1` | Yes | Yes | No gray skip terminates at Decoder 1 (S55--61). This agrees with M149--155 and M258--269. |
| Detail Branch `X -> T`, 32 channels at half resolution | Yes | Yes | The input-to-detail path and `T: 32 ch, H/2 x W/2` label appear at S70--73, consistent with M276--281. The exact choice of normalized versus unnormalized branch input remains unresolved in the manuscript (M283). |
| Detail projection `T -> T_hat` | No | No | M287--291 requires a separable-convolution projection before fusion. The SVG labels the branch output as `T` (S70--72), sends it directly into GDF (S79), but names `T_hat` inside the GDF input expression (S77). No node or arrow identifies the mandatory transformation. |
| GDF gate `G` and modulation `G odot T_hat` | Partial | Not independently verifiable | The GDF box identifies `Cat(D1, G odot T_hat)` (S75--79), consistent with M293--299, but no gate-generation or modulation edge is shown. This may be intentionally delegated to Fig. 4, but Figure 1 cannot verify the full GDF operation. |
| GDF output `F_fuse`, 128 channels at half resolution | No | Not verifiable | The outgoing GDF-to-Head arrow exists (S84), but neither `F_fuse` nor its `128 ch, H/2 x W/2` contract is labeled. This conflicts with the caption's statement that tensor size is identified at every principal fusion point (M161) and omits Table 2's fusion output (M135--138). |
| Final prediction `F_fuse -> z0_half -> Upsample(H,W) -> z0 -> sigmoid -> P` | Partial | No, as drawn | The SVG shows GDF to `Head` and states `z0 -> sigmoid -> P` (S81--85), so sigmoid placement is directionally correct. It omits both `z0_half` and `Upsample(H,W)`, which M309--315 require between the prediction convolution and the full-resolution logit. |
| Auxiliary heads from `D2`, `D3`, `D4` | Yes | Yes | Purple dashed arrows attach to the correct source features (S87--98), matching M317--323. Their required `Conv1x1 -> Upsample(H,W) -> z2/z3/z4` output paths are not labeled. |

## Complete arrow ledger

| # | SVG arrow | SVG location | Expected manuscript path | Verdict |
|---:|---|---|---|---|
| 1 | `Input X -> ConvNeXt Stage 1 (F1)` | S36 | `Encoder(X)` starts from `X` | **Consistent** |
| 2 | `F1 -> F2` | S37 | Hierarchical encoder stage progression | **Consistent** |
| 3 | `F2 -> F3` | S38 | Hierarchical encoder stage progression | **Consistent** |
| 4 | `F3 -> F4` | S39 | Hierarchical encoder stage progression | **Consistent** |
| 5 | `F4 -> Residual bottleneck + MSC` | S44 | `F4 -> B0 -> B` (M143--145; M185--218) | **Partial** — endpoints are compressed into one box; `B0` and `B` are absent. |
| 6 | `Residual bottleneck + MSC -> Decoder 4 + LRS-E` | S58 | `B -> Decoder4(B) -> U4`, then LRS-E with `F3` (M146, M263--264) | **Partial** — broadly ordered correctly, but `B` and `U4` are not identified and the concatenation order is not drawn. |
| 7 | `D4 block -> D3 block` | S59 | `D4 -> Decoder3 -> U3 -> LRSE3` (M147, M265--266) | **Consistent at block level** |
| 8 | `D3 block -> D2 block` | S60 | `D3 -> Decoder2 -> U2 -> LRSE2` (M148, M267--268) | **Consistent at block level** |
| 9 | `D2 block -> Decoder 1 (D1)` | S61 | `D2 -> Decoder1 -> D1` (M149, M269) | **Consistent** |
| 10 | `F3 -> Decoder 4 + LRS-E` skip | S63 | `Cat(U4,F3) -> LRSE4` (M146, M264) | **Consistent source/target; partial operation trace** |
| 11 | `F2 -> Decoder 3 + LRS-E` skip | S64 | `Cat(U3,F2) -> LRSE3` (M147, M266) | **Consistent source/target; partial operation trace** |
| 12 | `F1 -> Decoder 2 + LRS-E` skip | S65 | `Cat(U2,F1) -> LRSE2` (M148, M268) | **Consistent source/target; partial operation trace** |
| 13 | `Input X -> Detail Branch` | S73 | `T = DetailBranch(X)` (M150, M276--281) | **Consistent** |
| 14 | `D1 -> GDF` | S78 | `Cat(D1,T_hat)` for gate and final fusion (M293--299) | **Consistent at high level** |
| 15 | `Detail Branch T -> GDF` | S79 | `T -> T_hat -> GDF` (M287--299) | **Inconsistent/incomplete** — the mandatory projection to `T_hat` is absent. |
| 16 | `GDF -> Head` | S84 | `F_fuse -> prediction convolutions -> z0_half -> upsample -> z0` (M309--315) | **Partial** — edge is plausible, but `F_fuse` and the half-to-full-resolution logit transition are missing. |
| 17 | `D2 -> Auxiliary head 2` | S96 | `D2 -> Conv1x1 -> Upsample(H,W) -> z2` (M317--321) | **Consistent attachment; incomplete output trace** |
| 18 | `D3 -> Auxiliary head 3` | S97 | `D3 -> Conv1x1 -> Upsample(H,W) -> z3` (M317--322) | **Consistent attachment; incomplete output trace** |
| 19 | `D4 -> Auxiliary head 4` | S98 | `D4 -> Conv1x1 -> Upsample(H,W) -> z4` (M317--323) | **Consistent attachment; incomplete output trace** |

## Findings by severity

### MAJOR — Figure 1 caption claims four visible MSC branches that the SVG does not contain

- **Locations:** M157--161; M193--220; S41--44; S100.
- **Evidence:** The caption says Figure 1 “shows ... the four MSC branches” (M161). The SVG instead contains only the one combined `Residual bottleneck + MSC` rectangle and one incoming/outgoing arrow (S41--44). Its footer names the branch types (S100), but no branch nodes or branch arrows exist.
- **Why it matters:** A reader cannot inspect `B0 -> R -> {C1,C3,C5,Cg} -> Cat -> projection -> B`, including the residual addition, from this figure. The asserted caption content is false for the supplied SVG.
- **Required correction:** Either (a) expand Figure 1 to display `B0`, `R`, all four branches, concatenation, projection, residual addition, and `B`; or (b) revise the caption to say that MSC is represented as one high-level block and is expanded only in Figure 2.

### MAJOR — The Detail Branch-to-GDF input path skips the required `T_hat` projection

- **Locations:** M274--283; M285--299; M303--305; S70--79.
- **Evidence:** The methods require `T_hat = GELU(BN(SepConv_3x3(T)))` before gate estimation and before gated concatenation (M287--299). Figure 1 draws `T` directly to GDF (S72, S79), while the GDF label already refers to `T_hat` (S77).
- **Why it matters:** The diagram leaves ambiguous whether the projection belongs to the Detail Branch, GDF, or is omitted. This is an unshown learnable operation and prevents an exact check of the stated GDF inputs.
- **Required correction:** Insert a labeled `Detail projection: T -> T_hat` node/arrow before GDF, with `T_hat: 32 ch, H/2 x W/2`; or rename/extend the Detail Branch box so it explicitly outputs `T_hat` and then make the internal boundary unambiguous.

### MAJOR — The final logit-resolution transition is missing from Figure 1

- **Locations:** M111--113; M137--138; M151--155; M307--325; S81--85.
- **Evidence:** The final head is specified as `F_fuse -> z0_half (H/2 x W/2) -> Upsample_H,W -> z0 (H x W) -> sigmoid -> P` (M309--315). The SVG provides only `Head: z0, P` and `z0 -> sigmoid -> P` (S81--85).
- **Why it matters:** The missing operation is the only stated spatial restoration from the half-resolution fused feature to the full-resolution loss/metric logit. It also obscures the distinction between a logit and a probability map.
- **Required correction:** Draw or label the head as `F_fuse -> prediction convs -> z0_half (1 x H/2 x W/2) -> bilinear Upsample(H,W) -> z0 (1 x H x W) -> sigmoid -> P`. Clarify in M151 whether `PredictionHead` includes the upsampling operation.

### MAJOR — Tensor naming and dimensional contracts are incomplete at the bottleneck, final fusion, and prediction output

- **Locations:** M121--139; M140--155; M159--161; S41--44; S55--57; S75--85.
- **Evidence:** Table 2 specifies `B0`, `B`, `T_hat`, `F_fuse`, `z0_half`, and `z0` (M123--138). None is explicitly identified in the main SVG; `D1` lacks even the manuscript's provisional `XX` channel field (S55--57). The caption nevertheless states that tensor size is identified at each principal fusion point (M161).
- **Why it matters:** The GDF output-width contract (`128`) and the prediction head's input contract cannot be checked. The figure also cannot distinguish the MSC output from its residual-bottleneck input.
- **Required correction:** Label the missing tensors and their channel/spatial sizes, retaining `XX` visibly where the final implementation is not yet frozen. At minimum add `B0`, `B`, `D1: XX ch`, `T_hat: 32 ch`, `F_fuse: 128 ch`, `z0_half: 1 ch, H/2 x W/2`, and `z0: 1 ch, H x W`.

### MODERATE — The decoder/LRS-E composite boxes hide the required operation order

- **Locations:** M142--155; M226--250; M256--272; S46--68.
- **Evidence:** The methods distinguish `Decoder_l`, `U_l`, `Cat(U_l,F_l)`, and `LRSE_l` (M146--148; M263--268). Each SVG composite box receives both a main-path arrow and a skip arrow directly (S46--68).
- **Why it matters:** The connections are source-correct, but the graphic cannot establish that upsampling and decoder projection occur before concatenation and LRS-E. This is a traceability limitation, not a demonstrated wrong topology.
- **Required correction:** Add small internal ports/nodes or state in the Figure 1 caption that each green block performs `Decoder_l` first and then `LRSE_l(Cat(U_l,F_l))`; retain Figure 3 as the detailed operation schematic.

### MODERATE — Auxiliary-head source attachments are correct, but their full-resolution logits are not represented

- **Locations:** M307--325; S87--98.
- **Evidence:** Figure 1 correctly attaches purple dashed heads to `D2`, `D3`, and `D4` (S96--98), matching M317--323. It does not label `z2`, `z3`, `z4`, their 1 x 1 convolutions, or their `Upsample(H,W)` transitions.
- **Why it matters:** The training-only intent is clear, but the diagram does not verify that every auxiliary loss receives a full-resolution logit compatible with `Y`.
- **Required correction:** Add `z2/z3/z4: 1 x H x W` output labels, or include a concise `1x1 conv + Upsample(H,W)` notation inside each head.

### MINOR — The diagram omits operations that are acceptable at high level only if the caption is narrowed

- **Locations:** M161; S41--44; S46--54; S70--79; S100--101.
- **Evidence:** Figure 1 appropriately uses high-level boxes for ConvNeXt stages and decoder blocks, but its caption overstates the diagram's detail by saying that all principal fusion tensor sizes and MSC branches are shown.
- **Required correction:** After addressing the major items, make the caption explicitly distinguish (i) high-level modules shown in Figure 1 from (ii) internal operations delegated to Figures 2--4.

## Revision roadmap

1. **Blocker — correct Figure 1's computation graph.** Add the explicit `T -> T_hat` projection and the explicit `z0_half -> Upsample(H,W) -> z0 -> sigmoid -> P` path. Ensure node labels use the same symbols as M287--315.
2. **Blocker — reconcile Figure 1 and its caption for MSC.** Either render all four MSC branches and the `B0`/`B` residual transition, or state that Figure 1 collapses MSC and refer readers to Figure 2.
3. **High priority — complete tensor contracts.** Add labels for `B0`, `B`, `D1` width, `T_hat`, `F_fuse`, `z0_half`, and full-resolution `z0`. Use the Table 2 notation exactly.
4. **High priority — make composite decoder blocks auditable.** Show the internal sequence or add a legend statement that the green blocks implement the equations in M263--269 before accepting the corresponding skip.
5. **Medium priority — complete deep-supervision notation.** Mark each auxiliary output as an upsampled full-resolution logit (`z2`, `z3`, `z4`), while retaining purple dashed training-only arrows.
6. **Verification criterion.** Re-run this 19-arrow ledger after revision. It should contain one unambiguous source and destination for every named tensor transition, and every tensor in Table 2 that participates in a Figure 1 fusion/head path should be visible or explicitly declared internal to a named composite module.

## AI disclosure

This audit was generated with AI assistance from a read-only comparison of the specified Markdown manuscript and SVG source. It did not inspect implementation code or execute the model; implementation-level correctness remains unverified.
