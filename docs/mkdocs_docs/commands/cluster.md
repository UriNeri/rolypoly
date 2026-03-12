# Cluster

> Auto-generated draft from CLI metadata for `rolypoly cluster`.
> Expand this page with command-specific context, examples, and citations.

## Summary

Cluster sequences by ANI/AAI (centroid, connected-components, leiden)

## Description

Cluster sequences by pairwise ANI/AAI identity and coverage.

    Supports on-the-fly ANI computation from FASTA files (pyskani, blastn,
    mmseqs2, k-mer overlap) and import of pre-computed alignment tables
    (BLAST outfmt 6, CheckV ANI table, MMseqs2 output). Multiple
    clustering algorithms are available: centroid/greedy (default,
    CD-HIT/CheckV style), connected-components (union-find), and Leiden
    community detection.

    Similarity measures (per Vclust DOI:10.1038/s41592-025-02701-7):
      - identity (ANI): nucleotide identity over the aligned region
      - tani: total ANI, bidirectional length-weighted average
      - global_ani: identity over the full query length (ANI * AF)
    Use --similarity-measure to select which one --min-identity applies to.

    Default thresholds (95% identity, 85% target coverage) follow the
    MIUViG species-level vOTU standards.


    Examples:
      # Cluster a FASTA file with default pyskani + centroid
      rolypoly cluster -i contigs.fasta -o clusters.tsv

      # Use pre-computed BLAST edges with connected-components
      rolypoly cluster -i blast.out --input-type blast6 \
          --clustering-method connected-components -o clusters.tsv

      # Leiden clustering at genus level (70% identity, 0% AF)
      rolypoly cluster -i contigs.fasta --min-identity 70 \
          --min-target-coverage 0 --clustering-method leiden -o clusters.tsv

      # Fast k-mer-only clustering (no alignment)
      rolypoly cluster -i contigs.fasta --ani-backend kmer -o clusters.tsv

      # pyskani with k-mer prefilter (fewer pairwise comparisons)
      rolypoly cluster -i contigs.fasta --kmer-prefilter -o clusters.tsv

      # Cluster using tANI instead of ANI for thresholding
      rolypoly cluster -i contigs.fasta --similarity-measure tani -o clusters.tsv

      # Use presets to mimic other tools
      rolypoly cluster -i contigs.fasta --preset checkv -o clusters.tsv
      rolypoly cluster -i contigs.fasta --preset fast-ani -o clusters.tsv
      rolypoly cluster -i contigs.fasta --preset kmer-fast -o clusters.tsv

      # Override a single preset option (e.g. lower identity threshold)
      rolypoly cluster -i contigs.fasta --preset checkv --min-identity 90 -o clusters.tsv

## Usage

```bash
rolypoly cluster [OPTIONS]
```

## Options

- `--preset`: Apply a named preset that configures multiple options at once. Explicit CLI flags always override the preset. See the epilog below for details on each preset. (type: `CHOICE`)
- `-i`, `--input`: Input file: FASTA/FASTQ for on-the-fly ANI computation, or a pre-computed pairwise table (BLAST outfmt 6, CheckV ANI table, MMseqs2 easy-search output) (type: `FILE`; required; default: `Sentinel.UNSET`)
- `--input-type`: Type of input file. 'fasta' triggers on-the-fly ANI computation using the --ani-backend. The table formats expect pre-computed pairwise results. (type: `CHOICE`; default: `fasta`)
- `--ani-backend`: Backend for computing pairwise ANI when --input-type is fasta. 'pyskani' is fast and suitable for most use cases. 'blastn' uses NCBI BLAST (requires blastn on PATH). 'mmseqs' uses MMseqs2 easy-search (requires mmseqs on PATH). 'kmer' uses k-mer overlap coefficient (fast, approximate). (type: `CHOICE`; default: `pyskani`)
- `--clustering-method`: Clustering algorithm. 'centroid': greedy length-sorted (CD-HIT/CheckV style). 'connected-components': union-find transitive closure. 'leiden': Leiden community detection (requires igraph+leidenalg). (type: `CHOICE`; default: `centroid`)
- `--min-identity`: Minimum pairwise identity threshold (0-100 scale) (type: `FLOAT RANGE`; default: `95.0`)
- `--min-target-coverage`: Minimum target (shorter sequence) coverage threshold (0-100) (type: `FLOAT RANGE`; default: `85.0`)
- `--min-query-coverage`: Minimum query (longer sequence) coverage threshold (0-100) (type: `FLOAT RANGE`; default: `0.0`)
- `--min-alignment-fraction`: Minimum alignment fraction (min(qcov, tcov), 0-100). When > 0, overrides individual qcov/tcov thresholds. (type: `FLOAT RANGE`; default: `0.0`)
- `--min-alignment-length`: Minimum individual alignment length (for blast6 parsing) (type: `INTEGER RANGE`; default: `0`)
- `--min-evalue`: Maximum evalue for individual alignments (blast6 parsing / blastn) (type: `FLOAT`; default: `0.001`)
- `--mmseqs-sensitivity`: MMseqs2 sensitivity parameter (-s) when --ani-backend is mmseqs (type: `FLOAT RANGE`; default: `7.5`)
- `--leiden-resolution`: Resolution parameter for Leiden clustering (higher = more clusters) (type: `FLOAT RANGE`; default: `1.0`)
- `--fasta-lengths`: FASTA file for reading sequence lengths (used by centroid clustering for length-sorted ordering). Only needed when --input-type is not fasta. (type: `FILE`)
- `-o`, `--output`: Output path for per-sequence cluster assignments (type: `FILE`; default: `cluster_assignments.tsv`)
- `--summary-output`: Output path for cluster summary table (default: <output>.summary.<ext>) (type: `FILE`)
- `--edges-output`: Output path for the filtered edge table (default: not written; useful for inspection) (type: `FILE`)
- `--representatives-fasta`: Output FASTA of cluster representative sequences. Only available when --input-type is fasta or --fasta-lengths points to a FASTA file. (type: `FILE`)
- `--output-format`: Tabular output format for assignments and summary tables (type: `CHOICE`; default: `tsv`)
- `-t`, `--threads`: Number of threads for ANI computation backends (type: `INTEGER RANGE`; default: `4`)
- `--similarity-measure`: Which similarity column to threshold with --min-identity. 'identity' (=ANI): identity over the aligned region. 'tani': total ANI, bidirectional length-weighted. 'global_ani'/'global_ani_query': identity over full query length. Derived columns are computed automatically when chosen. (type: `CHOICE`; default: `identity`)
- `--kmer-prefilter`, `--no-kmer-prefilter`: Run a k-mer overlap prefilter before alignment-based ANI. Only sequence pairs passing the k-mer threshold are sent to the alignment backend. Ignored when --ani-backend is 'kmer'. (type: `BOOLEAN`; default: `False`)
- `--kmer-k`: K-mer length for the kmer backend or kmer prefilter (type: `INTEGER RANGE`; default: `15`)
- `--kmer-prefilter-threshold`: Minimum k-mer overlap coefficient (0-1) for prefilter pairs. Lower values retain more pairs (higher recall, slower). Only used when --kmer-prefilter is set. (type: `FLOAT RANGE`; default: `0.5`)
- `--flag-repeats`, `--no-flag-repeats`: Run a self-dotplot repeat check on every input sequence. Sequences whose longest internal repeat track spans more than --repeat-max-fraction of their length are flagged as potential assembly artefacts and excluded from representative selection (they are still clustered normally). Only available when the input is FASTA. (type: `BOOLEAN`; default: `True`)
- `--repeat-k`: K-mer size for the repeat-flag dotplot analysis (type: `INTEGER RANGE`; default: `15`)
- `--repeat-max-fraction`: Maximum fraction of sequence length covered by the longest repeat track before the sequence is flagged. Lower values flag more aggressively. (type: `FLOAT RANGE`; default: `0.4`)
- `--log-file`: Optional log file path (type: `FILE`)

## Additional Notes

```text
Presets (--preset NAME):
  Presets override multiple options at once to match common tool
  configurations.  Explicit CLI flags always take priority over
  preset values.

  cd-hit               CD-HIT-EST-style (mmseqs, 95% ANI, no AF filter, greedy centroid)
  checkv               CheckV anicalc+aniclust (blastn, 95% ANI, 85% AF, centroid)
  genus                Rough genus-level (blastn, 70% ANI, no AF, connected-components)
  kmer-fast            Kmer-db 2-style (k-mer overlap only, fast approximate, 90% identity)
  leiden-community     Leiden community detection (blastn, 90% ANI, resolution=1.0)
  miuvig-species       MIUViG species-level vOTU (95% ANI, 85% AF, blastn, centroid)
  mmseqs-cluster       MMseqs2 easy-cluster-style (mmseqs backend, 95% ANI, 85% AF)
  pyfastani            FastANI/pyfastani-style (tANI >= 95%, adjusted frag len, centroid)
  pyskani              skani/pyskani-style (tANI >= 95%, no AF filter, centroid)


Similarity measures (details from Vclust, DOI:10.1038/s41592-025-02701-7):
  identity / ANI    M(A,B) / L(A,B)
                    Nucleotide identity over the aligned region.
  tANI              (ANI1*AF1*LEN1 + ANI2*AF2*LEN2) / (LEN1+LEN2)
                    Total ANI, bidirectional length-weighted average.
  global_ani        ANI * AF = M(A,B) / |A|
                    Identity normalised by full sequence length.
  kmer_overlap      |kmers(A)∩kmers(B)| / min(|kmers(A)|,|kmers(B)|)
                    K-mer overlap coefficient (fast, approximate).

  AF(A->B)  = L(A,B) / |A|   alignment fraction (query coverage)
  AF(B->A)  ≈ L(B,A->B) / |B| (target coverage, approximated from
              target coordinates of the A->B alignment)
```


