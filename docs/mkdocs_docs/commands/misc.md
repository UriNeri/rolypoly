# Miscellaneous & Quality of Life Commands

This page documents utility commands in RolyPoly that help with data handling, sequence processing, and other common tasks. These are available under the `misc` group or as standalone commands.

---

<a id="shrink-reads"></a>
## Shrink Reads (`shrink-reads`)
Subsample FASTQ files by number or proportion.

**Usage:**
```bash
rolypoly shrink-reads -i input.fq -o output_dir --subset-type random --sample-size 10000
```
- `--subset-type`: `top_reads` (default) or `random`
- `--sample-size`: Number of reads (int) or proportion (float <1)

```mermaid
flowchart TD
	A["Input: FASTQ (.fq/.fastq, gz) "] --> B["Sample selection<br> (top_reads / random)"] --> C["Output: subsampled FASTQ<br> (.fq / .fq.gz)"]
	classDef io fill:#f0f9ff,stroke:#0366d6,color:#03396c;
	class A,C io
	class B fill:#fffaf0,stroke:#b85c00,color:#7a3b00;
```

---

<a id="mask-dna"></a>
## Mask DNA (`mask-dna`)
Mask viral-like sequences in a reference genome using various aligners.

**Usage:**
```bash
rolypoly mask-dna -i input.fa -o masked.fa -a mmseqs2 -r ref.fa
```
- `-a, --aligner`: Aligner backend (`minimap2`, `mmseqs2`, `diamond`, `bowtie1`, `bbmap`)
- `-r, --reference`: Custom masking reference
- `--tmpdir`: Temporary directory

```mermaid
flowchart TD
	M1["Input: Reference FASTA (.fa/.fasta)"] --> M2["Align to viral DB<br> (minimap2 / bbmap / mmseqs)"] --> M3["Mask regions (N / lowercase)"] --> M4["Output: masked FASTA"]
	class M1,M4 io
	class M2,M3 fill:#fffaf0,stroke:#b85c00,color:#7a3b00;
```

---

<a id="fastx-stats"></a>
## FASTX Stats (`fastx-stats`)
Calculate sequence statistics for FASTA/FASTQ files.

**Usage:**
```bash
rolypoly fastx-stats -i input.fa -o stats.tsv -f length -f gc_content -f n_count
```
- `--fields`: Choose which stats to report
- `--format`: Output format (`tsv`, `csv`, `md`)
- `-c, --circular`: Treat sequences as circular for analysis

```mermaid
flowchart TD
	S1["Input: FASTA/FASTQ"] --> S2["Parse sequences (polars)"] --> S3["Compute stats<br> (length, GC, N-count, hash)"] --> S4["Output: stats.tsv/csv/md/parquet"]
	class S1,S4 io
	class S2,S3 fill:#fffaf0,stroke:#b85c00,color:#7a3b00;
```

---

<a id="rename-sequences"></a>
## Rename Sequences (`rename-seqs`)
Standardize sequence IDs in a FASTA file and generate a mapping table.

**Usage:**
```bash
rolypoly rename-seqs -i input.fa -o renamed.fa -m mapping.tsv --prefix CID --hash
```
- `--prefix`: Prefix for new IDs
- `--hash/--no-hash`: Use hash instead of running number
- `--stats/--no-stats`: Include sequence stats in mapping

---

<a id="quick-taxonomy"></a>
## Quick Taxonomy (`quick-taxonomy`)
Fast taxonomic assignment for marker search results or contigs. (Experimental)

**Usage:**
```bash
rolypoly quick-taxonomy -i marker_results.tsv -o taxonomy.tsv
```
- `--marker_results`: Optional marker-search results file
- `--format`: Output format (`text`, `json`, `tsv`)
- `--min_score`: Minimum score for taxonomy assignment

---

<a id="fetch-sra"></a>
## Fetch SRA (`fetch-sra`)
Download SRA/ENA FASTQ files using ENA API and aria2c/wget.

**Usage:**
```bash
rolypoly fetch-sra -i SRR12345678 -o ./sra_downloads
```
- `-i, --input`: Run accession (e.g., `SRR...`) or file with accessions (one per line)
- `--report`: Also download XML report metadata
- Downloads all FASTQ files for a run ID
- Requires aria2c or wget installed

---

For more details on each command, use `rolypoly [COMMAND] --help`.
