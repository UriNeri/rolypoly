# Examples
Scenario-based examples for the most common RolyPoly workflows/commands.
Note - command help messages are updated on a more frequent basis, so for full option lists, run `rolypoly <command> --help`.

---
# `filter-reads` and `assemble`
## Background

### tl;dr preset choice

How your RNA was prepared for sequencing is usually the main thing that determines which preset to use. Three factors tend to matter most: 1. whether your library was **poly-A selected**, 2. whether you want to filter potential **host cDNA/mRNA** and other contaminants, and 3. whether your sample came from enriched/purified virions (**"virome"**), where you can often skip rRNA removal and sometimes use lighter quality filtering.

- If your library is poly-A selected: RolyPoly includes a poly-A trimming step in `filter-reads`.
- If you want host filtering: provide a host reference genome to `filter-reads` and `assemble` (RolyPoly will try to mask likely viral regions in that reference before using it for filtering).
- If your sample is virome-enriched: you can often skip rRNA and host removal, use lighter quality filtering, and sometimes enable deduplication steps (depending on your goals).

*Note: "host" removal can also remove EVEs or integrated viral sequences, so it may not be ideal if you are interested in non-RNA viruses or endogenous viral elements related to your study.*

---

### From sample to reads: the wet-lab pipeline

Before we get "reads", the biological material goes through a series of wet-lab steps that strongly shape what the sequencing data look like. Understanding that chain helps with picking in-silico settings and avoiding unexpected read loss.

The steps and methods below are usually chosen based on study goals and budget. Many introduce bias, but also reduce cost or increase throughput. For example, rRNA depletion kits do not equally target all domains of life, but they can substantially reduce the sequencing depth needed.

In most non-targeted genomics or metagenomics studies, the starting material can be anything from a single host species and its infecting virus, to a complex mixture spanning multiple domains of life (table 1). The "meta" prefix (e.g. metagenomics, metatranscriptomics) generally implies a mixed community as the source.

Regardless of source type, material is usually processed through a mix of chemical, enzymatic, and physical steps depending on study focus. The tables below summarize common steps across the wet-lab pipeline: sample handling before nucleic acid extraction (table 2), extraction itself (table 3), post-extraction processing (table 4), and library preparation (table 5). For RNA virus-centric/transcriptomic work, this usually means RNA extraction, often with rRNA depletion or poly-A selection, followed by reverse transcription to cDNA. Direct RNA sequencing is also possible (e.g. Oxford Nanopore), though still less common in many workflows.

Note: this overview is intentionally practical and not exhaustive.

**Table 1: Common sample sources for RNA virus discovery**

| Source | What it contains | Notes for virus discovery |
|---|---|---|
| Monoculture / host cells / tissue | Predominantly host RNA; viral RNA if infected | High host rRNA background expected. Consider including a host reference for filtering if target virus(es) can be distinguished from host sequences - note that some viruses have endogenized relatives (EVEs) or an integrated stage (e.g. retroviruses). |
| Environmental (water / soil / sediment / biofilm / atmospheric dust...) | Mixed community (bacteria, archaea, eukaryotes, viruses...) | No single host reference; rRNA dominates. |
| Environmental concentrate (e.g. filter retentate 0.2–3 µm, ultracentrifugation) | Similar to raw environmental sources, but pre-processed to enrich or deplete size fractions | Can be used for community-level expression analysis (metatranscriptomics) as well as virus discovery. Depending on pre-processing, free virions may be depleted, meaning virus discovery may reflect intracellular viral RNA rather than extracellular virions. |
| VLPs / protected environmental nucleic acids (e.g. ultracentrifugation / sucrose gradient / FeCl₃ precipitation / 0.1 µm filtrate / TFF) | Enriched sub-cellular fraction; virions, membrane vesicles, mobile genetic elements; some host debris | DNase/RNase treatment prior to extraction can greatly reduce host background, but low yields may necessitate amplification (e.g. RCA, MDA), which can introduce chimeras and skew abundances. |

---

Once the sample is collected, it usually goes through pre-extraction processing. These steps act on physical or chemical properties of the sample (size, density, protection status, membrane/protein integrity) and can strongly affect what ends up in the extracted material. For our use case, one key distinction is whether the sample was further enriched for viruses (e.g. VLP-focused workflows, often called "viromics"). Some steps also serve multiple practical purposes, such as buffer exchange, inhibitor removal, or transport/storage stabilization.

A practical note on filtration: the steps listed in table 2 can be used in combination, and it is entirely possible to use the retentate from one filter as the input to a second filtration step, or to collect the filtrate instead - the choice depends on which size fraction is of interest.

**Table 2: Common pre-nucleic acid extraction processing steps**

| Processing Step | What it does | Notes for virus discovery |
|---|---|---|
| Centrifugation (low speed / clarification) | Removes large debris and intact cells by pelleting | Standard first step in many protocols; supernatant is carried forward |
| Homogenization / bead beating | Disrupts tissue or cell-rich material into a homogenate | Releases intracellular contents including host RNA; increases host background |
| Filtration (large pore, e.g. 5–20 µm) | Removes large debris, most eukaryotic cells, and multicellular material | Helps reduce host cell material; virions and small prokaryotes pass through freely |
| Filtration (small pore, e.g. 0.2–0.45 µm) | Retains bacteria and larger particles; virions pass into filtrate | Can deplete prokaryotic cells; filtrate is enriched for virions and small VLPs |
| Tangential flow filtration (TFF) | Concentrates particles within a size range | Useful for large volume samples (e.g. seawater); retains virions while removing small molecules |
| Ultracentrifugation / sucrose gradient | Separates particles by size and density | Can enrich for VLP/virion fractions; may co-pellet host debris and membrane vesicles |
| Chemical treatment (e.g. FeCl₃ precipitation) | Precipitates virions or nucleic acid-associated particles | Low-cost concentration method for large volumes; efficiency varies by virus type |
| Proteinase K treatment | Degrades proteins in the sample | Can improve nucleic acid yield and purity by removing protein-nucleic acid complexes; may also disrupt non-enveloped virion capsids if not controlled carefully |
| Chloroform / organic solvent treatment | Denatures and removes lipids and proteins | Commonly used in phenol-chloroform extraction; also used pre-extraction to disrupt enveloped viruses and remove host membrane material - note this will also disrupt enveloped virions |
| DNase / RNase treatment (pre-extraction) | Degrades unprotected extracellular nucleic acids in the sample | Reduces free host nucleic acid contamination; nucleic acids inside virions or vesicles are protected and retained |

Many of these steps are combined within a single protocol or commercial kit. Some steps also serve multiple roles - for example, a buffer exchange step may simultaneously remove inhibitors while concentrating the sample.

---

Nucleic acid extraction is the step that liberates nucleic acids from biological material. The method you choose can introduce bias in yield, purity, and which nucleic acid species are recovered. For RNA virus discovery, RNA extraction is usually the default, though total nucleic acid (TNA) extraction is also used when both DNA and RNA are of interest (for example, mixed RNA/DNA virus surveys or integrated proviral sequence detection).

**Table 3: Common nucleic acid extraction methods**

| Method | What it recovers | Notes for virus discovery |
|---|---|---|
| TRIzol / phenol-chloroform extraction | Total RNA (and DNA if desired) | Robust and widely used; recovers a broad size range of RNA; requires careful handling of hazardous reagents; co-extracted DNA can be removed by DNase treatment if needed |
| Silica column-based RNA extraction (e.g. Qiagen RNeasy, Zymo) | Total RNA above a size threshold (typically >200 nt) | Convenient and fast; small RNAs (e.g. siRNAs, miRNAs) may be lost unless a specific small RNA kit is used; yields can be low for dilute or complex samples |
| Total nucleic acid (TNA) extraction | Both DNA and RNA | Useful when both RNA and DNA viruses are of interest; downstream DNase or RNase treatment can separate the pools if needed |
| Small RNA extraction / enrichment | RNA below ~200 nt (miRNA, siRNA, piRNA) | Specifically targets small RNA species; relevant for studies of RNA silencing or small RNA-based virus detection; full-length viral genomes will be depleted |
| Direct lysis (e.g. in low-input or single-cell protocols) | Variable; depends on lysis conditions | Minimises handling loss; may co-extract inhibitors; often paired with whole-transcriptome amplification |

---

After extraction, a second round of processing is often used to reshape the nucleic acid pool by removing unwanted species or adjusting yield/purity before library prep. These steps act on the molecules directly (not the physical sample) and can have a large effect on what gets sequenced.

**Table 4: Common post-nucleic acid extraction processing steps**

| Processing Step | What it does | Notes for virus discovery |
|---|---|---|
| DNase treatment (on-column or in-solution) | Degrades residual DNA from the extracted nucleic acid pool | Important for RNA-focused studies to reduce DNA carryover; critical if reverse transcription is used, to avoid amplifying DNA templates |
| Ribosomal RNA (rRNA) depletion | Removes rRNA using probe-based hybridization and depletion (e.g. RiboZero, NEBNext) | rRNA is often the dominant fraction in total RNA (commonly around 80-90%+); depletion increases the proportion of informative transcripts, but probe sets may not cover non-model organisms well |
| Poly-A selection | Captures polyadenylated RNA using oligo-dT beads | Enriches for eukaryotic mRNA and some RNA viruses that have poly-A tails; depletes rRNA, bacterial RNA, and non-polyadenylated viral RNA |
| Size selection | Retains RNA fragments within a target size range | Can be used to enrich for small RNAs such as siRNAs or miRNAs; may deplete full-length viral genomes |
| RNA concentration / cleanup | Removes salts, enzymes, or other inhibitors (e.g. column cleanup, ethanol precipitation) | Important for downstream enzymatic steps; low-yield samples risk loss during cleanup |
| Amplification (e.g. RCA, MDA) | Increases the total amount of nucleic acid | Necessary for low-yield samples; can introduce chimeric reads, amplification bias, and skewed abundances - complicates quantification |

---

Library preparation converts processed nucleic acid into sequencing-ready molecules. For RNA libraries, this begins with reverse transcription to cDNA; after that, the workflow broadly resembles DNA library prep. Choices at this stage, especially fragmentation, priming strategy, and strand specificity, can strongly affect how viral sequences are represented and interpreted downstream.

When working with multiple samples (for example, time points, locations, or tissue types), it helps to plan sample-level barcoding/indexing before library prep starts. Each sample can get a unique index (or dual-index combination), so libraries can be pooled in one sequencing run and demultiplexed later. This is standard and usually lowers per-sample cost. But prep decisions still matter: if samples are pooled too early (for example at the RNA stage), sample identity is lost. Keeping a clear index-to-sample map is essential for meaningful downstream comparisons. In practice, indexing strategy is often constrained by the sequencing center to avoid collisions with other projects in the same run. Many users therefore receive already-demultiplexed data; if not, you may also see index FASTQs in addition to R1/R2 files.

**Table 5: Common library preparation steps**

| Step | What it does | Notes for virus discovery |
|---|---|---|
| Reverse transcription (RT) | Converts RNA to cDNA using reverse transcriptase | Required for RNA sequencing on most short-read and some long-read platforms; primer choice (random hexamers, oligo-dT, or gene-specific) affects coverage and representation. In non-targeted approaches, sequence-specific primers shouldn't be used |
| Second-strand synthesis | Converts single-stranded cDNA to double-stranded cDNA (dsDNA) | Enables ligation-based library preparation; strand information may be lost depending on the method used |
| RNA/cDNA fragmentation | Breaks nucleic acid into shorter fragments suitable for sequencing | Fragmentation method (chemical, enzymatic, heat, or sonication) can affect coverage uniformity and introduce biases; may be performed on RNA prior to RT or on cDNA afterward |
| End repair & A-tailing | Blunts fragment ends and adds an adenosine overhang | Prepares fragments for adapter ligation; standard step in most Illumina library prep workflows |
| Adapter ligation | Ligates platform-specific adapters to fragment ends | Adapters contain primer binding sites and optional barcodes/indices; ligation efficiency affects library complexity |
| Indexing / barcoding | Adds unique index sequences to each sample's library | Enables multiplexing of multiple samples in a single sequencing run; dual indexing reduces index-hopping artifacts. See note above on sample tracking strategy |
| PCR amplification | Amplifies the adapter-ligated library | Excess PCR cycles reduce library complexity and introduce duplicates - particularly problematic for quantification of viral abundance |
| Size selection (post-ligation) | Selects fragments within a target insert size range (e.g. 150–500 bp) | Removes adapter dimers and very short/long fragments; affects read length distribution and insert size |
| Strand-specific / directional library prep | Preserves information about which strand the RNA was transcribed from | Important for correctly orienting viral RNA segments, identifying antisense transcription, and reducing ambiguity in de novo assemblies |

#### From sample to FASTQ: the processing chain

```mermaid
flowchart TD
    A[Sample\ncells / retentate / VLP pellet] --> B[RNA extraction\ne.g. TRIzol, column-based]
    B --> C{Enrichment or depletion?}
    C -->|Total RNA + ribo-depletion\nRibo-Zero, RNase H-based| D[ribo-depleted RNA\n★ recommended for broad virus discovery]
    C -->|Poly-A selection\noligodT capture| E[poly-A-enriched mRNA]
    C -->|No treatment\ntotal RNA| F[total RNA\nhigh rRNA content]
    D & E & F --> G[RNA fragmentation\nenzymatic or physical shearing\ntarget ~150–400 bp]
    G --> H[Reverse transcription\nrandom hexamers or oligo-dT]
    H --> I[End repair + dA-tailing]
    I --> J[Adapter ligation\nP5 + P7 Illumina adapters]
    J --> K[PCR amplification\n6–15 cycles]
    K --> L[Size selection\nSPRI beads or gel\nsets insert size distribution]
    L --> M[Paired-end sequencing\ntypically 2×75 bp or 2×150 bp]
    M --> N[Raw FASTQ\nR1 + R2 per sample]
```

  ### From digitized data to in-silico processing

  > **Important:** the goal of in-silico processing is not to remove as many reads as possible.
  > The goal is to reduce technical noise and increase biological signal for your specific question.

  This part is inherently goal-dependent. Trimming, filtering, masking, host subtraction, normalization,
  and deduplication are tools, not goals by themselves. If a step does not clearly improve signal for the
  biological target you care about, it is often better to keep it conservative or skip it.

  Pipeline, tool, and parameter choices should follow the study target, but they could also be informed by
  everything that happened before digitization (sample source, enrichment strategy, extraction chemistry,
  library prep, expected contaminants, expected insert size, and sequencing platform).

  Examples:

  - If you also care about DNA viruses, avoid aggressive host/DNA removal.
  - If you know exactly which adapter sequence was used, start with that specific adapter set instead of broad default databases (some of which may match real viral sequences).
  - If your focus is differential expression (host), preserve host reads and avoid filtering choices that distort host transcript abundance.
  - If your focus is low-level variation (for example SNPs or quasispecies/subspecies structure), avoid error correction, over-normalization, and aggressive deduplication that can flatten minor variants.
  - If your focus is recovering major viruses and improving genome completeness, stronger cleanup and downsampling reads with over represented kmers may actually make assembly easier.

  It is also completely fine to run multiple in-silico "branches" from the same sample, each with different pipeline choices.
  For example, you can use the original raw reads (or lightly processed reads) for quantification,
  while using a normalized subset for assembly-oriented recovery.

  Many recent workflows also combine strategies, such as pooling selected samples,
  running multiple assemblers and merging/curating outputs, applying k-mer abundance-based read normalization,
  and in some cases adding haplotyping/strain-resolution analyses after the fact.

### Common in-silico steps mapped to `filter-reads`

The list below follows a common read-processing order and maps each stage to the internal
`filter-reads` step names. Each step includes examples for how to skip it (`--skip-steps`),
when it is skipped by presets, and/or how to tune behavior with `--override-parameters`.

#### Decision flow for read-processing branches

```mermaid
flowchart TD
    A[Raw FASTQ] --> B{Need host or known DNA removal?}
    B -->|yes| C[filter_known_dna]
    B -->|no| D[skip filter_known_dna]

    C --> E{Need rRNA depletion in silico?}
    D --> E
    E -->|yes| F[decontaminate_rrna]
    E -->|no| G[skip decontaminate_rrna]

    F --> H{Use identified-DNA filtering?}
    G --> H
    H -->|yes| I[filter_identified_dna]
    H -->|no| J[skip filter_identified_dna]

    I --> K[dedupe]
    J --> K

    K --> L[trim_adapters]
    L --> M{Poly-A library?}
    M -->|yes| N[trim_polya_tails]
    M -->|no| O[skip trim_polya_tails]

    N --> P[remove_synthetic_artifacts]
    O --> P
    P --> Q[entropy_filter]

    Q --> R{Short inserts / overlap expected?}
    R -->|yes| S[error_correct_1]
    S --> T[error_correct_2]
    T --> U[merge_reads]
    R -->|no| V[skip overlap-heavy steps]

    U --> W[quality_trim_unmerged]
    V --> W
    W --> X[final dedupe + outputs]

    X --> Y{Single branch or multiple branches?}
    Y -->|single| Z[One downstream path]
    Y -->|multiple| AA[Branch e.g. quantification vs assembly]
```

#### 1) Known DNA filtering: `filter_known_dna`

Used for host or known DNA contaminant subtraction when you provide `-D/--known-dna`.
If `--known-dna` is not provided, this step is automatically skipped.

```bash
# Use a custom known DNA reference
rolypoly filter-reads -i reads/ -o filtered/ \
  -D host_or_contaminant.fasta

# Skip known DNA filtering explicitly
rolypoly filter-reads -i reads/ -o filtered/ \
  --skip-steps filter_known_dna

# Tune matching strictness
rolypoly filter-reads -i reads/ -o filtered/ -D host.fasta \
  --override-parameters '{"filter_known_dna": {"k": 31, "mincovfraction": 0.8, "hdist": 0}}'
```

#### 2) rRNA decontamination: `decontaminate_rrna`

Uses packaged rRNA references (SILVA + NCBI masked sets).

```bash
# Default run (with rRNA filtering)
rolypoly filter-reads -i reads/ -o filtered/

# Skip rRNA filtering
rolypoly filter-reads -i reads/ -o filtered/ \
  --skip-steps decontaminate_rrna

# Also skipped by this preset
rolypoly filter-reads -i reads/ -o filtered/ \
  --preset all_virus_metag

# Make rRNA filtering stricter/looser
rolypoly filter-reads -i reads/ -o filtered/ \
  --override-parameters '{"decontaminate_rrna": {"mincovfraction": 0.7, "k": 31}}'
```

#### 3) Identified DNA filtering: `filter_identified_dna`

This step uses the rRNA stats profile to fetch candidate genomes and filter likely host/DNA reads.

```bash
# Run with identified-DNA filtering
rolypoly filter-reads -i reads/ -o filtered/

# Skip identified-DNA filtering
rolypoly filter-reads -i reads/ -o filtered/ \
  --skip-steps filter_identified_dna

# Already skipped by these presets
rolypoly filter-reads -i reads/ -o filtered/ --preset fast
rolypoly filter-reads -i reads/ -o filtered/ --preset all_virus_metat
rolypoly filter-reads -i reads/ -o filtered/ --preset all_virus_metag

# Tune filtering sensitivity
rolypoly filter-reads -i reads/ -o filtered/ \
  --override-parameters '{"filter_identified_dna": {"mincovfraction": 0.8, "k": 31}}'
```

#### 4) Deduplication (early pass): `dedupe`

`dedupe` appears here in the main processing chain, and another dedupe pass is run at final output stage.

```bash
# Skip the early dedupe stage in the main chain
rolypoly filter-reads -i reads/ -o filtered/ \
  --skip-steps dedupe

# Tune dedupe aggressiveness
rolypoly filter-reads -i reads/ -o filtered/ \
  --override-parameters '{"dedupe": {"passes": 1, "s": 0}}'

# strict preset increases dedupe aggressiveness
rolypoly filter-reads -i reads/ -o filtered/ --preset strict
```

#### 5) Adapter trimming: `trim_adapters`

Adapter trimming runs after early decontamination and before quality trimming.

```bash
# Skip adapter trimming (usually not recommended)
rolypoly filter-reads -i reads/ -o filtered/ \
  --skip-steps trim_adapters

# Tune adapter trim behavior
rolypoly filter-reads -i reads/ -o filtered/ \
  --override-parameters '{"trim_adapters": {"k": 23, "mink": 11, "hdist": 1, "minlen": 20}}'

# If you want to use only a known custom adapter set, pre-trim externally,
# then skip internal adapter trimming in rolypoly.
cutadapt -a file:my_adapters.fa -A file:my_adapters.fa \
  -o pretrim_R1.fq.gz -p pretrim_R2.fq.gz reads_R1.fq.gz reads_R2.fq.gz

rolypoly filter-reads -i pretrim_R1.fq.gz,pretrim_R2.fq.gz -o filtered/ \
  --skip-steps trim_adapters
```

#### 6) Poly-A tail trimming: `trim_polya_tails`

Useful for poly-A selected libraries; disabled by default unless preset/flag enables it.

```bash
# Enable poly-A tail trimming explicitly
rolypoly filter-reads -i reads/ -o filtered/ --trim-polya

# Also enabled by this preset
rolypoly filter-reads -i reads/ -o filtered/ --preset poly_a_selected

# Skip poly-A trimming
rolypoly filter-reads -i reads/ -o filtered/ \
  --skip-steps trim_polya_tails

# Tune poly-A trimming
rolypoly filter-reads -i reads/ -o filtered/ --trim-polya \
  --override-parameters '{"trim_polya_tails": {"trimpolya": 18, "minlen": 20}}'
```

#### 7) Synthetic artifact filtering: `remove_synthetic_artifacts`

Targets synthetic/control artifacts.

```bash
# Skip synthetic artifact filtering
rolypoly filter-reads -i reads/ -o filtered/ \
  --skip-steps remove_synthetic_artifacts

# Tune k-mer matching
rolypoly filter-reads -i reads/ -o filtered/ \
  --override-parameters '{"remove_synthetic_artifacts": {"k": 31}}'
```

#### 8) Low-complexity filtering: `entropy_filter`

Filters very low-complexity reads (mostly homopolymers / low-entropy sequence).

```bash
# Skip entropy filtering
rolypoly filter-reads -i reads/ -o filtered/ \
  --skip-steps entropy_filter

# Tune entropy thresholds
rolypoly filter-reads -i reads/ -o filtered/ \
  --override-parameters '{"entropy_filter": {"entropy": 0.01, "entropywindow": 30}}'
```

#### 9) Overlap-based correction stage 1: `error_correct_1`

Most useful when paired reads overlap (short inserts). On single-end input, this step is auto-skipped.

```bash
# Skip stage 1 correction
rolypoly filter-reads -i reads/ -o filtered/ \
  --skip-steps error_correct_1

# fast preset already skips error_correct_1
rolypoly filter-reads -i reads/ -o filtered/ --preset fast

# Tune stage 1 behavior
rolypoly filter-reads -i reads/ -o filtered/ \
  --override-parameters '{"error_correct_1": {"mix": "t", "ordered": "t"}}'
```

#### 10) Correction stage 2: `error_correct_2`

Second correction stage. This one is not auto-skipped for single-end by default.

```bash
# Skip stage 2 correction
rolypoly filter-reads -i reads/ -o filtered/ \
  --skip-steps error_correct_2

# fast preset already skips error_correct_2
rolypoly filter-reads -i reads/ -o filtered/ --preset fast

# Tune stage 2 behavior
rolypoly filter-reads -i reads/ -o filtered/ \
  --override-parameters '{"error_correct_2": {"passes": 1, "reorder": true}}'
```

#### 11) Overlap merge: `merge_reads`

Merges overlapping pairs. Auto-skipped for single-end input.

```bash
# Skip merge stage
rolypoly filter-reads -i reads/ -o filtered/ \
  --skip-steps merge_reads

# Tune merge behavior
rolypoly filter-reads -i reads/ -o filtered/ \
  --override-parameters '{"merge_reads": {"mix": "f"}}'
```

#### 12) Quality trimming of unmerged reads: `quality_trim_unmerged`

Late-stage quality trimming after correction/merge steps.

```bash
# Skip quality trimming
rolypoly filter-reads -i reads/ -o filtered/ \
  --skip-steps quality_trim_unmerged

# Tune trim stringency
rolypoly filter-reads -i reads/ -o filtered/ \
  --override-parameters '{"quality_trim_unmerged": {"trimq": 12, "minlen": 25}}'
```

#### Final output stage note

After the main chain above, `filter-reads` runs a final dedupe pass on merged/interleaved outputs.
  

#### Key decisions that affect your RolyPoly preset

- **ribo-depletion vs. poly-A selection** - a major choice that can affect RNA virus recovery
  (see section below for evidence and caveats).
- **Insert size** - set by the size-selection step. Typical targets are 150–400 bp for Illumina.
  Short inserts (< 2×read length) produce overlapping read pairs; very short inserts bleed into
  adapter sequence.
- **Amplification method** - standard PCR introduces PCR duplicates. SISPA or random-primed
  amplification (used in some VLP protocols) introduces additional chimeric read artifacts
  (Kugelman et al., 2017).

---

### Library preparation and viral recovery

Many RNA viruses, including phages and most negative-sense, ambisense, and segmented RNA viruses, are not
polyadenylated, so enrichment strategy can strongly affect what gets recovered.
In a direct comparison, poly(A)-selected libraries yielded viral reads but were insufficient for complete
recovery of a non-polyadenylated virus genome, while ribo-depleted total RNA performed substantially
better (Visser et al., 2016; PMID: 27250973). In practice, **total RNA plus ribo-depletion is often a
good starting point** for discovery-oriented workflows.

Ribo-depletion also has limits. Probe performance depends on sequence match, so depletion can be less
effective in non-model or mixed communities (Kim et al., 2019; PMID: 31783730). Computational rRNA
filtering is therefore commonly used as a second layer (Zhou et al., 2018; PMID: 29444661), but very
aggressive filtering can discard informative reads. Dovrolis et al. showed that rRNA-focused sorting can
capture rRNA-virus chimeras and reduce viral support for assembly (PMID: 34370725).

Independent RNA-seq evidence also supports treating host-virus chimeras cautiously: host-virus chimeric
events can be infrequent and largely artifactual, consistent with RT template switching during library
construction (Yan et al., 2021; PMID: 33980601). For preprocessing, this argues for conservative
thresholds and careful interpretation of putative chimeric reads instead of blanket removal based on
single-feature matches.

*Subjective note:* I have also observed some rRNA-virus chimeric contigs in published virus discovery
works and general metatranscriptomes (see [Table S6, sheet "rRNA_summary" in Neri et al., 2022](https://www.sciencedirect.com/science/article/pii/S0092867422011187?via%3Dihub#mmc6)), and the full extent
of host-virus chimera is probably underappreciated. I am not sure I can recommend keeping chimeric
reads though - I assume these are not biological and often arise from template switching after nucleic
acid extraction - and properly trimming only the host portion of chimeric reads is not trivial (IMO).

RolyPoly preprocessing currently focuses on **Illumina short-read libraries** (poly-A selected, ribo-
depleted, or unselected total RNA). Long-read/direct-RNA support is a separate ongoing area.

---

### Sequencing artifacts and their computational signatures

Each step between nucleic acid extraction and the sequencer can introduce artifacts.

#### Paired-end reads and insert size

The **insert** is the actual cDNA fragment between the two Illumina adapters. Both ends are
sequenced, producing R1 (reads the forward strand) and R2 (reads the reverse complement strand).

```
Adapter-P5 ─── INSERT (cDNA fragment) ─── Adapter-P7
    │                                           │
    └──► R1 reads →                 ← R2 reads ◄┘

Typical layout (insert longer than read length, no overlap):

5'─[P5]─[R1 ████████████████]─ · · · gap · · · ─[████████████████ R2]─[P7]─3'
         read 1 (fwd)           unsequenced middle           read 2 (rev-comp)
         └────────────────────────────────────────────────────────────────────┘
                              insert size (e.g. 300 bp)
```

Insert size is set by the size-selection step in library prep. Typical Illumina runs use
2×75 bp or 2×150 bp reads; inserts are usually 150–400 bp. The ratio of insert to read length
determines which artifacts appear.

#### Short inserts, overlapping reads, and adapter contamination

As insert size decreases relative to read length, two problems arise:

```
Case 1 - Overlapping reads (insert slightly shorter than 2× read length):

5'─[P5]─[R1 ████████████]─[████ overlap ████]─[████████████ R2]─[P7]─3'

R1: ████████████████████                        (reads left → right)
R2:             ████████████████████            (reads right → left, RC)
              ←── overlap region ──►

BBMerge can merge overlapping pairs into a single, longer, more accurate read.
This is common for short viral genomes where short inserts are preferred.
```

```
Case 2 - Adapter contamination (insert shorter than read length):

5'─[P5]─[fragment]─[P7]─3'   (very short insert)

R1: [fragment sequence ████][P7-adapter sequence ░░░░░░░░░░]
                              ↑ adapter "bleed-in"
R2: [fragment RC ████][P5-adapter RC ░░░░░░░░░░]

Without adapter trimming, these tails can look like spurious sequence and reduce assembly quality.
Adapter trimmers (e.g., BBDuk, cutadapt, Trimmomatic) typically remove these by matching known
adapter sequence k-mers, and most also let you provide custom adapter sequences.
```

#### PCR duplicates vs optical duplicates

Both appear as multiple reads with identical sequence, but their causes differ:

```
PCR duplicates                    Optical duplicates
──────────────────────────────    ──────────────────────────────
Same molecule amplified           Single cluster on the flowcell
multiple times during             that spreads or is misread as
library prep PCR:                 two adjacent clusters:

Flowcell surface:                 Flowcell surface:

 ● cluster A  (seq: ACGTACGT)      ●●  cluster A+B touching
 ● cluster B  (seq: ACGTACGT)          (seq: ACGTACGT for both)
 ● cluster C  (seq: ACGTACGT)
 (scattered anywhere on tile)     ← physically adjacent, same tile

Cause: amplification bias         Cause: optical/diffusion artifact
Worse with: more PCR cycles,      Worse with: patterned flowcells
  low-input / VLP libraries         (NovaSeq, NextSeq 2000)
Removed by: sequence identity     Removed by: sequence identity
  (any duplicate-removal tool)      AND cluster proximity
```

Both can inflate abundance estimates and mislead assembly coverage. In metatranscriptomes, though,
highly expressed transcripts are expected, so aggressive duplicate removal can also discard real
biology. In RolyPoly, deduplication is done as exact-sequence removal (`seqkit rmdup`) rather than
coordinate-based deduplication.

#### Virus–host chimeric reads

Template switching during reverse transcription, or ligation of fragmented RNA molecules during
library prep, can join viral and host RNA into a single read:

```
Origin molecules:

  ─────[viral genomic RNA segment]─────3'
                                        ↕ RT jumps (template switching)
  5'─────[host mRNA ─────────────]─────

Resulting chimeric cDNA / read:

  5'─[viral sequence ███████████][host sequence ░░░░░░░░░░░]─3'
      maps to viral genome ─┘         maps to host ────┘

Consequence 1: chimeric read is filtered by host-removal step
               → viral portion is lost along with the host portion

Consequence 2: chimeric read enters assembly
               → chimeric contig spanning virus + host
               → false annotations or missed virus

Consequence 3: rRNA–virus chimeras (described in Dovrolis et al., 2021; PMID: 34370725)
               → rRNA-sorting tools bin the chimera as rRNA
               → virus reads lost; assembly of complete genome fails
```

This is one reason RolyPoly keeps a **permissive rRNA filter by default**: strict rRNA filtering can
discard the viral component of mixed/chimeric reads (Dovrolis et al., 2021; PMID: 34370725).
The same caveat applies to host subtraction, where very strict mapping thresholds may remove reads that
only partially match host sequence.

#### Practical implications for preset choice

- If broad RNA virus recovery is the priority, a practical starting point is total RNA with ribo-depletion and moderate rRNA filtering.
- If libraries are multiplexed on patterned-flowcell instruments, non-redundant dual indexing can help reduce index-swap cross-talk (Costello et al., 2018; PMID: 29739332).
- If inserts are short and read overlap is high, read merging often improves effective read quality and assembly input (Bushnell et al., 2017; PMID: 29073143).
- If low-input amplification was used, it is usually safer to handle duplicate/chimera filtering conservatively and validate candidates by remapping support (Kugelman et al., 2017; PMID: 28182717).

---

### Background references (PubMed-checked)

- Visser M, Bester R, Burger JT, Maree HJ. Next-generation sequencing for virus detection: covering all the bases. Virol J. 2016;13:85. doi:10.1186/s12985-016-0539-x. PMID: 27250973.
- Dovrolis N, Kassela K, Konstantinidis K, et al. ZWA: Viral genome assembly and characterization hindrances from virus-host chimeric reads; a refining approach. PLoS Comput Biol. 2021;17(8):e1009304. doi:10.1371/journal.pcbi.1009304. PMID: 34370725.
- Yan B, Chakravorty S, Mirabelli C, et al. Host-Virus Chimeric Events in SARS-CoV-2-Infected Cells Are Infrequent and Artifactual. J Virol. 2021;95(15):e00294-21. doi:10.1128/JVI.00294-21. PMID: 33980601.
- Neri U, Wolf YI, Roux S, et al. Expansion of the global RNA virome reveals diverse clades of bacteriophages. Cell. 2022;185(21):4023-4037.e18. doi:10.1016/j.cell.2022.08.023. PMID: 36174579.
- Kim IV, Ross EJ, Dietrich S, et al. Efficient depletion of ribosomal RNA for RNA sequencing in planarians. BMC Genomics. 2019;20(1):909. doi:10.1186/s12864-019-6292-y. PMID: 31783730.
- Zhou Q, Su X, Jing G, Chen S, Ning K. RNA-QC-chain: comprehensive and fast quality control for RNA-Seq data. BMC Genomics. 2018;19(1):144. doi:10.1186/s12864-018-4503-6. PMID: 29444661.
- Costello M, Fleharty M, Abreu J, et al. Characterization and remediation of sample index swaps by non-redundant dual indexing on massively parallel sequencing platforms. BMC Genomics. 2018;19(1):332. doi:10.1186/s12864-018-4703-0. PMID: 29739332.
- Bushnell B, Rood J, Singer E. BBMerge - Accurate paired shotgun read merging via overlap. PLoS One. 2017;12(10):e0185056. doi:10.1371/journal.pone.0185056. PMID: 29073143.
- Kugelman JR, Wiley MR, Nagle ER, et al. Error baseline rates of five sample preparation methods used to characterize RNA virus populations. PLoS One. 2017;12(2):e0171333. doi:10.1371/journal.pone.0171333. PMID: 28182717.
- Adiconis X, Borges-Rivera D, Satija R, et al. Comparative analysis of RNA sequencing methods for degraded or low-input samples. Nat Methods. 2013;10(7):623-629. doi:10.1038/nmeth.2483. PMID: 23685885.
- He S, Wurtzel O, Singh K, et al. Validation of two ribosomal RNA removal methods for microbial metatranscriptomics. Nat Methods. 2010;7(10):807-812. doi:10.1038/nmeth.1507. PMID: 20852648.

---

### Preset quick-reference

| `roll` preset | Library preparation | Filter preset | Assembly preset |
|---|---|---|---|
| `rna_virus` (default) | RNA virus metatranscriptome: rRNA removal, host + identified-DNA filter | `rna_virus_metat` | `rna_virus` |
| `ribodepleted` | Total RNA ribo-depleted: stricter rRNA removal (mincovfraction=0.7) | `total_rna_ribodepleted` | `rna_virus` |
| `poly_a` | Poly-A selected mRNA: polyA tail trim, stricter quality trim | `poly_a_selected` | `metatranscriptome` |
| `all_virus_metat` | All-virus metatranscriptome / RNA virome: relaxed rRNA filter, skips identified-DNA filter | `all_virus_metat` | `rna_virus` |
| `DNA_virus` | DNA virome / metagenomics: skips rRNA and identified-DNA filtering | `all_virus_metag` | `metag` (metaSPAdes only) |
| `complete` | Any — maximum sensitivity; runs all three assembler modes | `rna_virus_metat` | `complete` |
| `fast` | Any — quick preview; skips error correction and identified-DNA filter | `fast` | `fast` |

When unsure, start with `--preset rna_virus`. Check read-count retention in the log before committing
to a full run on many samples.

---

## End-to-end pipeline (`roll`)

The `roll` command runs the full discovery pipeline in one call:
read filtering → assembly → contig filtering → marker search → nucleotide search → annotation.
Pick the `--preset` that matches your library preparation.

### Viral RNA metatranscriptome (default)

Ribo-depleted total RNA from an environmental sample; expected to contain RNA viruses.
This is the default preset, so `--preset rna_virus` can be omitted.

```bash
rolypoly roll \
  --input reads_R1.fq.gz,reads_R2.fq.gz \
  --output-dir rp_out/ \
  --threads 16 --memory 64g \
  --preset rna_virus
```

### With host/contaminant removal

Provide a FASTA of the host genome (or any expected DNA contaminant) with `-D`.
The assembly step will also filter contigs that match the host.

```bash
rolypoly roll \
  --input reads_R1.fq.gz,reads_R2.fq.gz \
  --output-dir rp_out/ \
  --threads 16 --memory 64g \
  --preset rna_virus \
  --host host_genome.fasta
```

### Total RNA, ribo-depleted library

Use `ribodepleted` for stricter rRNA removal (mincovfraction=0.7).

```bash
rolypoly roll \
  --input reads/ \
  --output-dir rp_out/ \
  --threads 16 --memory 64g \
  --preset ribodepleted
```

### Poly-A selected mRNA library

Enables polyA tail trimming and uses rnaSPAdes+MEGAHIT assembly.

```bash
rolypoly roll \
  --input reads_R1.fq.gz,reads_R2.fq.gz \
  --output-dir rp_out/ \
  --threads 16 --memory 64g \
  --preset poly_a
```

### DNA virome / metagenomics (no rRNA or identified-DNA filtering)

Suitable for DNA-based viromes or metagenomic libraries where you do not want rRNA or identified-DNA filtering applied.  
NOTE: this isn't really rolypoly forte, this preset is just for convenience if you want to run a general pipeline that WON'T remove DNA data, or harm assembly of potential hosts.


```bash
rolypoly roll \
  --input reads/ \
  --output-dir rp_out/ \
  --threads 16 --memory 64g \
  --preset DNA_virus
```

### Maximum sensitivity

Runs all three assembler modes (metaSPAdes + rnaviralSPAdes + MEGAHIT).
Slowest, but highest chance of recovering divergent or low-abundance viruses.

```bash
rolypoly roll \
  --input reads_R1.fq.gz,reads_R2.fq.gz \
  --output-dir rp_out/ \
  --threads 32 --memory 128g \
  --preset complete
```

### Quick preview with `--mini`

Subsamples the input before running the pipeline; forces the `fast` assembly preset.
Useful for a rapid sanity-check before committing to a full run.

```bash
rolypoly roll \
  --input reads_R1.fq.gz,reads_R2.fq.gz \
  --output-dir rp_preview/ \
  --threads 8 --memory 16g \
  --preset rna_virus \
  --mini
```

### Override individual sub-presets

Use `--filter-preset` and/or `--assembly-preset` to mix-and-match independently of `--preset`.

```bash
# Strict read filtering, but fast assembly
rolypoly roll \
  --input reads/ \
  --output-dir rp_out/ \
  --filter-preset strict \
  --assembly-preset fast
```

### Resume a partial run
WARNING ! THIS IS NOT FULLY TESTED YET. Use at your own risk.
`--skip-existing` skips any step whose output directory already exists.

```bash
rolypoly roll \
  --input reads_R1.fq.gz,reads_R2.fq.gz \
  --output-dir rp_out/ \
  --preset rna_virus \
  --skip-existing
```

---

## Read filtering (`filter-reads`)

### Paired-end reads, default settings

```bash
rolypoly filter-reads \
  -i reads_R1.fq.gz,reads_R2.fq.gz \
  -o filtered_reads/ \
  -t 8 -M 16g
```

### Directory of FASTQ files

All FASTQ files in the directory are processed; paired files are matched by base name.

```bash
rolypoly filter-reads \
  -i raw_reads/ \
  -o filtered_reads/ \
  -t 8
```

### With a preset

```bash
# RNA virus metatranscriptome (lenient quality trim, rRNA removal at mincovfraction=0.6)
rolypoly filter-reads -i reads/ -o filtered/ --preset rna_virus_metat

# Total RNA ribo-depleted (stricter rRNA removal mincovfraction=0.7)
rolypoly filter-reads -i reads/ -o filtered/ --preset total_rna_ribodepleted

# Poly-A selected library (enables polyA trimming, stricter quality trim)
rolypoly filter-reads -i reads/ -o filtered/ --preset poly_a_selected

# All-virus metatranscriptome (relaxed rRNA filter, skips identified-DNA filter)
rolypoly filter-reads -i reads/ -o filtered/ --preset all_virus_metat

# All-virus metagenomics (skip rRNA + identified-DNA filters entirely)
rolypoly filter-reads -i reads/ -o filtered/ --preset all_virus_metag

# Fast (skip error correction and identified-DNA filter)
rolypoly filter-reads -i reads/ -o filtered/ --preset fast
```

### With host removal

```bash
rolypoly filter-reads \
  -i reads_R1.fq.gz,reads_R2.fq.gz \
  -o filtered_reads/ \
  -D host_genome.fasta \
  --preset rna_virus_metat
```

### Override a specific step parameter

```bash
# Raise rRNA coverage threshold and use a stricter quality trim
rolypoly filter-reads \
  -i reads/ -o filtered/ \
  --preset rna_virus_metat \
  --override-parameters '{"decontaminate_rrna": {"mincovfraction": 0.8}, "quality_trim_unmerged": {"trimq": 15}}'
```

### Skip specific steps

```bash
# Run everything except error correction
rolypoly filter-reads \
  -i reads/ -o filtered/ \
  --skip-steps error_correct_1 --skip-steps error_correct_2
```

---

## Assembly (`assemble`)

### From a directory of filtered reads, default settings

```bash
rolypoly assemble \
  -id filtered_reads/ \
  -o assembly_out/ \
  -t 16 -M 64g
```

### With a preset

```bash
# RNA virus: rnaviralSPAdes + MEGAHIT, broad k-mer range, rmdup post-processing
rolypoly assemble -id filtered_reads/ -o assembly_out/ --preset rna_virus

# Metatranscriptome: rnaSPAdes + MEGAHIT
rolypoly assemble -id filtered_reads/ -o assembly_out/ --preset metatranscriptome

# Metagenomics: metaSPAdes (for DNA-based libraries)
rolypoly assemble -id filtered_reads/ -o assembly_out/ --preset metag

# Fast: MEGAHIT only, narrow k-mer range
rolypoly assemble -id filtered_reads/ -o assembly_out/ --preset fast

# Complete: all three assembler modes
rolypoly assemble -id filtered_reads/ -o assembly_out/ --preset complete
```

### Explicit library specification

```bash
# Paired-end
rolypoly assemble \
  --paired-end 1 reads_R1.fq.gz reads_R2.fq.gz \
  -o assembly_out/ -t 8 -M 32g

# Multiple libraries mixed
rolypoly assemble \
  --paired-end 1 lib1_R1.fq.gz lib1_R2.fq.gz \
  --merged 2 lib2_merged.fq.gz \
  -o assembly_out/ -t 8 -M 32g
```

### Run only rnaviralSPAdes

```bash
rolypoly assemble \
  -id filtered_reads/ \
  -o assembly_out/ \
  -A spades_rnaviral
```

### Override k-mer settings

```bash
rolypoly assemble \
  -id filtered_reads/ -o assembly_out/ \
  --preset rna_virus \
  --override-parameters '{"megahit": {"k-min": 27, "k-max": 99, "k-step": 12}}'
```

### Skip post-processing deduplication

```bash
rolypoly assemble \
  -id filtered_reads/ -o assembly_out/ \
  --preset rna_virus \
  --skip-steps post_processing
```

---

## Modular step-by-step workflow

The `roll` command chains these steps automatically, but each can be run independently
for finer control or to slot into an existing pipeline.

```bash
# 1. Filter reads
rolypoly filter-reads \
  -i raw_reads/ -o filtered_reads/ \
  -t 16 -M 32g --preset rna_virus_metat

# 2. Assemble
rolypoly assemble \
  -id filtered_reads/ -o assembly/ \
  -t 16 -M 64g --preset rna_virus

# 3. Filter assembled contigs against a host reference
rolypoly filter-contigs \
  -i assembly/final_assembly.fasta \
  --host host_genome.fasta \
  -o assemblies/filtered_assembly.fasta \
  -t 8

# 4. Search for viral marker genes (RdRps, genomad)
rolypoly marker-search \
  -i assemblies/filtered_assembly.fasta \
  -o marker_results/ \
  -t 16

# 5. Nucleotide-level search against known RNA virus databases
rolypoly virus-mapping \
  -i assemblies/filtered_assembly.fasta \
  -o virus_hits.tab \
  -t 16

# 6. Annotate candidate contigs
rolypoly annotate \
  -i assemblies/filtered_assembly.fasta \
  -o annotation/ \
  -t 16
```

---

## Marker gene search (`marker-search`)

```bash
# Search all supported databases (RdRp HMMs + geNomad)
rolypoly marker-search \
  -i contigs.fasta -o marker_out/ -t 8

# Search only the RdRp database
rolypoly marker-search \
  -i contigs.fasta -o marker_out/ --database rdrp -t 8
```

## Virus nucleotide mapping (`virus-mapping`)

```bash
rolypoly virus-mapping \
  -i contigs.fasta \
  -o virus_hits.tab \
  -t 8
```

## Shrink / subsample reads

```bash
# Random subsample to 50 000 reads (for a quick test)
rolypoly shrink-reads \
  -i reads_R1.fq.gz,reads_R2.fq.gz \
  -o sampled/ \
  --subset-type random --sample-size 50000

# Coverage-normalise with bbnorm (better for paired data)
rolypoly shrink-reads \
  -i reads_R1.fq.gz,reads_R2.fq.gz \
  -o sampled/ \
  --subset-type bbnorm
```
