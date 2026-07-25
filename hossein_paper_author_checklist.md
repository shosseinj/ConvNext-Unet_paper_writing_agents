# Author Verification Checklist for the Revised Manuscript

This checklist accompanies `hossein_paper_revised.md` and `hossein_paper_revised.docx`. It records items that cannot be resolved from the supplied Word draft alone. No experimental result has been invented.

## Blocking Items

- [ ] Confirm the final title, author list, affiliations, corresponding-author email, funding, and author contributions.
- [ ] Freeze one code version, configuration file, checkpoint, and experiment identifier.
- [ ] Remove every `XX`, `[REF VERIFY]`, `NR`, and author placeholder before submission.
- [ ] Confirm that the intended paper is the ConvNeXt-Tiny U-Net paper. The source file also contained an unrelated Boundary-Frequency Injection Module draft.
- [ ] Reconcile the Kvasir-SEG split. The source reports 900 training images and 500 test images from a 1000-image dataset, which is impossible without an additional corpus or a different counting convention.
- [ ] Publish exact split lists, validation IDs, random seed, duplicate-screening rule, and image-level or video-level grouping policy.
- [ ] Recover the exact loss function, auxiliary weights, sigmoid placement, smoothing constants, scheduler, maximum epochs, early-stopping patience, checkpoint criterion, and augmentation pipeline from code and experiment logs.
- [ ] Generate one canonical result ledger. Each result must link to a dataset split, checkpoint, seed, metric script, input size, and post-processing setting.

## Originality and Attribution

- [ ] Do not retain the exact title "Sharpening Lightweight Models for Generalized Polyp Segmentation: A Boundary Guided Distillation from Foundation Models" from the source draft. It matches Agnihotri et al., arXiv:2604.17865, which proposes LiteBounD.
- [ ] Do not retain the source draft's BFIM, FFD, Sobel, frequency-splitting, cross-attention, or MFAF block unless the authors can document its independent design and cite its sources.
- [ ] Do not present the source draft's `LiteBoundD` row or the copied CTNet/MEGANet table values as original experiments. The revised manuscript labels them as provisional `Pub. draft` values; replace them with independently evaluated or traceable values before submission.
- [ ] Use `LRS-E` (Learned Local-Response Skip Enhancement) for the proposed module. The old `BSEI` acronym collides with imported source material and is not retained.
- [ ] Recreate all figures from the final code and test predictions. Do not copy MPDF, SSFA, MDI, LiteBounD, HCA-Net, or other external diagrams or heatmaps.
- [ ] Run a similarity check after the final rewrite and manually inspect titles, module names, captions, tables, and figure labels.

## Conflicts in the Supplied Draft

The following values were not selected in the revised manuscript because they conflict:

| Item | Conflicting source values | Required action |
|---|---|---|
| Model name | Ours, Our, Ours-2, Our9388, Ours-93.04, Ours-93.66 | Use one name: Proposed ConvNeXt-Tiny U-Net |
| Parameters | 38.5 M; 29.61 M; 29,609,302; 29.60 M | Recount the frozen inference graph and report the convention |
| GFLOPs | 14.57 G in one table; missing elsewhere | Recompute at a stated input size and counting tool |
| Kvasir-SEG mDice/mIoU | 0.9240/0.8608; 0.9349/0.8777; 0.9258/0.8641; 0.9389/0.8866; 0.935/0.879 | Trace each value to a checkpoint or remove it |
| CVC-ClinicDB mDice/mIoU | 0.9519/0.9084; 0.9531/0.9104; 0.951/0.906 | Trace each value to a checkpoint or remove it |
| CVC-ColonDB mDice/mIoU | 0.8126/0.7044; 0.7972/0.6916; 0.7974/0.6920 | Trace each value to a checkpoint or remove it |
| ETIS mDice/mIoU | 0.8739/0.7761; 0.8869/0.7967; 0.887/0.797 | Trace each value to a checkpoint or remove it |
| CVC-300 mDice/mIoU | 0.9149/0.8432; 0.9161/0.8452; 0.916/0.845 | Trace each value to a checkpoint or remove it |
| Training duration | About 100, 120, 1370, 1405, and 2836 | Identify whether each number is an epoch, iteration, or run identifier |
| CVC-ColonDB total | 380 in the draft; other protocols use different counts | Confirm the exact corpus version and test list |
| Metric scale | Percentages such as 79.74 and fractions such as 0.797 | Use one scale throughout the paper |

## Provisional Values Now Used

The revised manuscript uses the most complete decimal result set from the supplied draft as provisional point estimates. These values are not independently verified and have `XX` for unavailable variability or boundary metrics:

| Dataset | mDice | mIoU | F_beta^w | S_alpha | mE_phi | maxE_phi | MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Kvasir-SEG | 0.935 | 0.879 | 0.901 | 0.935 | 0.952 | 0.963 | 0.025 |
| CVC-ClinicDB | 0.951 | 0.906 | 0.931 | 0.954 | 0.983 | 0.990 | 0.009 |
| CVC-300 (EndoScene) | 0.916 | 0.845 | 0.848 | 0.939 | 0.951 | 0.969 | 0.007 |
| CVC-ColonDB | 0.7974 | 0.6920 | 0.759 | 0.869 | 0.885 | 0.903 | 0.035 |
| ETIS-LaribPolypDB | 0.887 | 0.797 | 0.730 | 0.873 | 0.883 | 0.913 | 0.014 |

- [ ] Confirm that this single-run result set came from the proposed ConvNeXt-Tiny U-Net, not HCA-Net or another legacy model.
- [ ] Confirm the source checkpoint, split, epoch/iteration, threshold, metric code, and input size for every row.
- [ ] Confirm the CVC-ColonDB conversion from 79.74/69.20 percent to 0.7974/0.6920.
- [ ] Replace each `XX` variability field with repeated-run statistics or remove the claim of repeated runs.
- [ ] Add verified boundary F-score and HD95 values, or remove those columns from the final submission version.

## Method Verification

- [ ] Confirm that the final architecture has exactly three LRS-E skip stages: `F3`, `F2`, and `F1`.
- [ ] Confirm that `D1` has no encoder skip connection and record its channel count.
- [ ] Confirm every tensor shape in Table 1, including `A_l` channel behavior and broadcasting.
- [ ] Confirm the MSC global branch output size and broadcast operation.
- [ ] Confirm whether the Detail Branch consumes normalized or unnormalized input.
- [ ] Confirm all dropout rates, normalization layers, padding, activation order, stochastic-depth schedule, and layer-scale initialization.
- [ ] Confirm the ConvNeXt-Tiny pretrained-weight source and fine-tuning policy.
- [ ] Confirm whether auxiliary heads are included in parameter and FLOP counts.
- [ ] Confirm binary-logit output, sigmoid placement, mask resizing, and inference threshold.
- [ ] Confirm whether any test-time augmentation, post-processing, or connected-component filtering is used.

## Dataset and Evaluation Verification

- [ ] Provide the exact number of images in train, validation, and test splits for every dataset.
- [ ] State whether the 145-image validation set is removed from the training pool or is counted within the nominal training total.
- [ ] Confirm the source and rights for every dataset and describe the ETIS source accurately.
- [ ] Screen for duplicate and near-duplicate frames before split generation.
- [ ] Report image and mask interpolation methods and mask binarization threshold.
- [ ] Define whether mDice and mIoU are image-wise means or global confusion-matrix scores.
- [ ] Define `epsilon`, inference threshold, probability calibration, and output resolution.
- [ ] Implement and cite S-measure, weighted F-measure, E-measure, and MAE with one verified codebase.
- [ ] Add boundary F-score at a stated tolerance and HD95 or ASSD if boundary claims are retained.
- [ ] Run at least **XX** independent seeds and report mean +/- standard deviation or confidence intervals.

## Ablation and Efficiency Verification

- [ ] Run the cumulative baseline -> MSC -> LRS-E -> DB -> GDF -> DS sequence.
- [ ] Run full-model leave-one-out controls for MSC, LRS-E, and DS.
- [ ] Compare GDF with direct concatenation and addition under the same training budget.
- [ ] Include parameter count and GFLOPs for every ablation variant.
- [ ] Report boundary metrics or contour analysis for the LRS-E ablation.
- [ ] Measure batch-1 latency, FPS, peak memory, precision mode, and hardware for the final model and selected baselines.
- [ ] Do not call the model lightweight or real-time unless the measurements support those terms.

## Figures and Tables

The source document contains five embedded PNG assets. They should not be inserted into the revised manuscript without verification:

- `image2.png` is a useful six-image polyp example figure and may be retained as a representative dataset figure after checking source attribution and image rights.
- `image1.png` labels attention maps as MPDF, SSFA, and MDI. These labels do not match MSC, LRS-E, or GDF and should not be used for this manuscript.
- `image3.png` contains feature labels such as `Down_stage1`, `Up_stage5`, and a table headed HCA. It appears to belong to a different model version.
- `image4.png` compares HCA-Net, CAFE-Net, and Poly-LVT. It is not a verified result for the proposed model.
- `image5.png` compares an `Our` model with CTNet, MEGANet, CAFE-Net, PraNet, CFANet, Polyp-PVT, U-Net++, and U-Net. The split, checkpoint, and method implementations are not documented.

The revised manuscript now includes original vector methodology figures for:

- Figure 1: complete network architecture, including all skip paths, auxiliary heads, and `z0_half -> z0 -> P` operations.
- Figure 2: MSC module.
- Figure 3: LRS-E and GDF module overview.
- Figure 4: Detail Branch and GDF close-up.
- Figure 5: dataset and evaluation workflow.

The following data-dependent figures remain intentionally ungenerated because the final model predictions and feature tensors were not supplied:

- Figure 6: qualitative segmentation comparison.
- Figure 7: learned local-response and gate maps.
- Figure 8: feature progression through the network.
- Figure 9: size-stratified results and failure cases.
- Tables 1-10: tensor dimensions, state of the art, datasets, training settings, primary results, comparisons, ablations, and complexity.

Each placeholder caption states what the final figure or table must contain. Replace the placeholders with figures generated from the final code and verified test predictions.

## Citation Verification

- [ ] Replace the source's `[ho-*]`, `[20-1]`, `[47-1]`, `[48-1]`, `[25-*]`, `[34-2]`, `[26-2]`, `[29-2]`, and empty citation keys with the numbered reference list in the revised manuscript.
- [ ] Correct the source's dataset paragraph, which cites PraNet as `[8]`; PraNet is Reference [7] in the revised list.
- [ ] Verify the final venue and metadata for TransUNet and HarDNet-MSEG; the revised list retains explicit `[REF VERIFY]` markers for these records.
- [ ] Confirm the corrected Polyp-PVT, ConvNeXt, PraNet, Structure-measure, video-segmentation benchmark, WM-DOVA, CTNet, MEIN, CIFFormer, and PFPRNet metadata against publisher records.
- [ ] Confirm the MEGANet DOI and page range; indexed records conflict between 7970-7979 and 7985-7994.
- [ ] Verify the added state-of-the-art records [24-40], especially LACINet, ECTransNet, Polyper, Boundary Refinement Network, CAFE-Net, PolyMamba-Net, MLB-Net, HCA-Net, DVFIP-Net, LiteBounD, and BDKD-Net.
- [ ] Keep LiteBounD and BDKD-Net as cited overlap comparators only; do not reuse their foundation-model or distillation concepts as contributions of this paper.
- [ ] Verify that all state-of-the-art values marked `Pub. draft` are linked to the original paper's table, split, input size, metric code, and publication year.
- [ ] Verify the complete author lists for Kvasir-SEG, EndoScene, and all recent comparator papers.
- [ ] Add DOI and final issue information where available.
- [ ] Do not use competitor numbers copied from a different split as direct evidence of superiority.
- [ ] Retain `[REF VERIFY]` markers until a reference manager or publisher record confirms the metadata.

## Submission Gate

- [ ] All placeholder values are resolved.
- [ ] All claims in the abstract match the final result tables.
- [ ] All table and figure numbers are sequential and cited in the text.
- [ ] Every result has a provenance record.
- [ ] Every baseline is either reproduced under the same protocol or clearly labeled as a published, non-direct comparison.
- [ ] Discussion claims are limited to the evidence.
- [ ] Data, code, ethics, funding, competing-interest, author-contribution, and generative-AI declarations are complete.
- [ ] The target Elsevier journal's guide for authors has been checked for word count, reference style, graphical abstract, highlights, data statement, and AI disclosure.
- [ ] A final human author review has verified the scientific content, code, figures, tables, references, and all `XX` replacements.
