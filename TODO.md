# RolyPoly TODOs

Updated: 2026-09-22. 

Unchecked items remain open. **Partial** means a useful implementation exists but the stated follow-up remains. **Confirm scope/status** means a maintainer decision is pending; it is not a claim that the feature is absent.

## Release and installation

- [x] Finish reviewing and committing the ORFfinder, installation-badge and
  BBTools-comment follow-ups before pushing.
- [ ] Publish the intended GitHub/PyPI release; keep the standalone environment's
  exact RolyPoly version pin aligned with that release.
- [ ] Submit the Bioconda dependency update as a new PR, with the released source
  version/checksum and appropriate build number. Keep its existing upstream
  import/CLI smoke tests; additional review tests stay outside the recipe.
- [ ] Build the actual proposed conda package and validate its installation.
  Dependency solving, wheel installation, 41 focused tests and recipe CLI smoke
  checks passed; an actual recipe build and full suite in that environment have
  not yet been completed.
- [ ] Track dependency builds needed for a complete Python 3.14 conda installation
  (currently blocked by pyfastani). Package metadata allows Python 3.11–3.14;
  this is not a claim that every interpreter/channel combination was tested.
- [ ] Verify that the reference-data release at
  [Zenodo DOI 10.5281/zenodo.21639933](https://doi.org/10.5281/zenodo.21639933)
  matches the current development data, including contents and checksums, and
  align download instructions. The maintainer supplied this DOI; its freshness
  has not yet been verified.
- [ ] Wait for a BBTools release containing the paired-reader fix, update the
  bbmapy bundle, and validate both BBNorm passes before removing the workaround.
  The tested stock 40.02 binary still reproduced the defect.

## Reads and assembly

- [ ] Add optional persistent `.fastq.gz` outputs for reads discarded by host
  and rRNA filtering. Inspection found no `--keep-host`, `--keep-rna` or
  `--keep-rrna` option. Proposed names: `--keep-host` and `--keep-rrna` (rRNA,
  rather than all RNA). Both steps use BBDuk: use its `outm`/`outm2` outputs,
  preserve pairing, keep categories separate, and register outputs so temporary
  cleanup cannot remove them. Keep this opt-in and expose it through `roll` too.
  BBDuk's matched output can include other active filter failures; describe it
  as reads discarded at that step, not confirmed host/rRNA sequences. Retaining
  reads discarded by other QC stages would require additional output handling.
- [ ] **Partial:** develop and benchmark low-memory presets. Existing memory
  controls and `shrink-reads` do not establish a validated low-memory workflow.
  Evaluate alternatives such as deacon only if needed.
- [ ] **Partial:** validate long-read filtering and define supported ONT/PacBio
  workflows. Single-end handling and SPAdes hybrid inputs already work; dedicated
  long-read assemblers such as MetaFlye/MetaMDBG remain candidates, not integrations.
- [ ] **Partial:** define a dedicated co-assembly strategy beyond pooling supplied
  inputs in `assemble`; update or retire the incomplete `co_assembly.py` workflow.
- [ ] **Partial:** evaluate contig cross-assembly/extension on biological fixtures,
  including user-supplied supplemental sequences (e.g. RACE-derived sequence).
  `extend` is implemented and has tests; broader validation remains separate.
- [ ] Develop and validate strain de-entanglement/refinement. Keep this separate
  from contig extension; forwarding reads to a variant caller is not sufficient.
- [ ] **Deferred / design decision:** poly(A) genome termini rescue by mapping
  reads from before poly(A) trimming back to contigs. Also consider mapping-based
  termini rescue when no poly(A) trimming was performed. Decide whether this
  belongs in `extend`, `termini`, or a precursor step, and define the read inputs
  and evidence needed to support a rescued terminus.
- [ ] Flag potential internal gene truncation caused by scaffolding: detect
  internal poly(N) stretches and examine read evidence to distinguish Ns present
  in raw reads from gaps apparently introduced by the assembler. Consider using
  mapped reads and other contigs to investigate alternative assembly paths.
  Where rescue is unsupported, retain the uncertainty and warn users through
  GFF/sequence features and/or a report table, including affected genes.
  Consider a dedicated red-ribbon track for poly(N) stretches in genome maps.

## Protein and RNA annotation

- [ ] Add peptidase/proteinase cleavage motif scanning or prediction.
- [ ] Add ORF 5′ UTR analysis, including candidate ribosome-binding/Kozak features
  and the limits of using these features to infer a host.
- [ ] **Deferred:** dedicated IRES detection and validation remain open; generic
  Rfam searches do not close this task, and `detect_ires` is disabled. Candidates
  include irespy, deepires and potentially
  [IRESGet](https://github.com/BioBurning/IRESGet). Their large dependency overhead
  rules out integration for now.
- [ ] Implement or validate TR-loop detection for leviviruses; the source lists
  disagree between “To Do” and “In Progress”, with no completion evidence.
- [ ] Add ribosomal slippage detection.
- [ ] Add ribosome-shunting prediction.
- [ ] **Partial:** audit GFF output consistency across commands, including feature
  IDs, coordinates, provenance and optional embedded sequence. Protein coordinate
  maps exist; embedded FASTA in GFF3 must use `##FASTA`, not `//FASTA`.
- [ ] **Partial:** audit remaining hit-consolidation and interval-operation modes
  against examples and tests. Core selection/overlap operations are implemented;
  do not mark every mode validated from the core tests alone.

## Genome interpretation and reporting

- [ ] **Partial:** maintain and validate the existing `report` command and report
  generation from `roll`. Track retained/discarded read counts and optional
  discarded-read output paths in reporting as those outputs are implemented;
  verify reports remain useful when optional pipeline steps are skipped.
- [ ] **Partial:** develop and validate circularity assessment. Terminal-repeat
  detection in `termini` is useful evidence, not proof of biological circularity.
- [ ] Add genome-completeness estimation.
- [ ] Add virion-type inference: capsid/capsid-less, lipid envelope and symmetry,
  with clear uncertainty and provenance.
- [ ] Add segmentality inference. If initially taxonomy-based, state its limits
  for incomplete genomes and unclassified sequences.
- [ ] Add strandedness inference, initially only where taxonomy supports it;
  retain an unknown category.
- [ ] Add host prediction, initially broad host domain where justified. Evaluate
  sequence-based methods and tools such as vhamster; do not imply an integration exists.
- [ ] Evaluate additional phenotype prediction, including the earlier Kallisto
  model suggestion, before choosing a method or promising support.
- [ ] **Confirm scope:** decide whether `roll` needs a single packaged results
  archive. Output folders and the interactive report already exist; the older
  TODO explicitly mentioned a tar.gz bundle.
- [ ] Extend visualization examples where needed (e.g. BAM/GFF workflows);
  external viewer integrations remain optional enhancements to the existing report.

## Benchmarking and documentation

- [ ] Benchmark end-to-end runtime, peak memory and practical minimum resources
  on representative datasets. Run statistics and targeted hash/read-filter
  experiments are not a complete resource benchmark.
- [ ] Compare discovery/assembly results with relevant tools (earlier candidates:
  AliMarko and Hecatomb), with defined datasets and fair configurations.
- [ ] Define appropriate sensitivity, specificity, precision/recall and other
  performance metrics for each benchmark; do not apply every metric indiscriminately.
- [ ] Expand end-to-end use cases and tutorials; keep CLI-generated option blocks
  aligned with Python definitions while preserving scientific/manual explanations.
- [ ] Document supported input layouts, biological assumptions and limitations
  alongside runnable examples as new workflows are added.
- [ ] Create a demo of using RolyPoly results in `suv-tk`, showing the required
  output files and steps to reproduce the example.
- [ ] Add more `report.html` examples and consider a visual interpretation guide
  with annotated screenshots explaining tracks, tables, warnings and uncertainty.

## Completed or superseded

- [x] Provide rRNA read filtering using BBDuk with NCBI/SILVA references. The
  maintainer confirmed filtering satisfies the original rRNA task; this does not
  establish a separate RNA-annotation or contamination-quantification workflow.
- [x] Close automatic RAM-disk management as not planned: expected dataset sizes
  and laptop/workstation users make it impractical as a general workflow.
  Existing `--temp-dir` allows user-selected storage; `--keep-tmp` only retains
  temporary files and does not create a RAM disk.
- [x] Use the additional virus-oriented Prodigal models through `pyrodigal-rv`.
- [x] Provide RNA secondary-structure prediction with LinearFold and RNAfold;
  their runtime interfaces have smoke tests.
- [x] Support single-sample assembly with native mate inputs, orphan/merged reads,
  filter-reads folder handoff, contig ID normalization and output mapping.
- [x] Use SPAdes dataset manifests; pool metaSPAdes paired-library file lists with
  a warning, preserving mate order without concatenating input files.
- [x] Support PenguiN native pairs and mixed-input streaming, including raw FASTA
  as ordinary sequences; explicitly document pairing loss in mixed mode.
- [x] Preserve explicit parameter overrides, correct assembly skip behavior,
  restore CASAVA warnings/comments and standardize the `first_n` subset name.
- [x] Align coverage defaults, examples and scientific background; explain that
  ribodepleted describes experimental treatment. The separate coverage experiment
  did not justify changing the current thresholds.
- [x] Implement CLI/config regression testing and database-aware smoke scenarios
  in `src/tests`; supersedes the old `utils/test.py` testing reference.
- [x] Implement rank-aware taxonomy assignment in `mmtax` with tests.
- [x] Implement interactive reporting and genome maps via `report`,
  `commands/virotype/summary.py` and `utils/viz/genome_maps.py`; supersedes the
  old visualization placeholder and unnamed duplicate task.
- [x] Publish a Bioconda package. Updating it for current dependencies remains an
  open release task above.
- [x] Declare clustering dependencies for PyPI/Pixi installs and support xxhash
  3.7.0 with explicit seed 0 and documented fallback consistency limits.
- [x] Restore standalone mamba installation via `src/setup/env_big.yaml`, including
  release-version pinning. Pixi and this YAML supersede the `quick_setup.sh` task.
- [x] Replace the ad hoc ORFfinder shell download with shared discovery,
  platform-aware official-binary installation and execution verification.
- [x] Add Pixi and standalone-mamba installation badges and instructions.

## Pending maintainer answers

1. Should a single packaged results archive (the historical tar.gz task) remain
   planned in addition to output folders and the existing report?

Task owners and historical priority disagreements are intentionally not guessed.
The release work is grouped first; the remaining scientific backlog needs its
own prioritization rather than inheriting conflicting “v1 MVP” labels.
