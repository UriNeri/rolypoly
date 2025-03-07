# End to End Pipeline

`end-2-end` executes the complete RolyPoly workflow from raw reads to virus identification.

## Options

### Common
- `-i, --input`: Raw RNA-seq data (fastq/gz file or directory) (required)
- `-o, --output-dir`: Output directory (default: current_directory_rp_e2e)
- `-t, --threads`: Number of threads (default: 1)
- `-M, --memory`: Memory allocation (default: "6g")
- `-D, --host`: Host/contamination fasta file
- `--keep-tmp`: Keep temporary files (flag)
- `--log-file`: Path to log file (default: current_directory/rolypoly_pipeline.log)

### Assembly
- `-A, --assembler`: Assemblers to use (default: "spades,megahit,penguin")
- `-d, --post-cluster`: Enable clustering (flag)

### Filtering
- `-Fm1, --filter1_nuc`: First nucleic filter (default: "alnlen >= 120 & pident>=75")
- `-Fm2, --filter2_nuc`: Second nucleic filter (default: "qcov >= 0.95 & pident>=95")
- `-Fd1, --filter1_aa`: First amino acid filter (default: "length >= 80 & pident>=75")
- `-Fd2, --filter2_aa`: Second amino acid filter (default: "qcovhsp >= 95 & pident>=80")
- `--dont-mask`: Skip masking RNA virus-like sequences in host fasta (flag)
- `--mmseqs-args`: Additional MMseqs2 arguments
- `--diamond-args`: Additional Diamond arguments (default: "--id 50 --min-orf 50")

### Marker Gene Search
- `--db`: Database to use (default: "all")

## Pipeline Steps

1. Read Filtering
   - Quality control
   - Host/DNA removal
   - rRNA decontamination

2. Assembly
   - Multi-assembler processing
   - Optional clustering
   - Final assembly generation

3. Assembly Filtering
   - Nucleotide/protein filtering
   - Host sequence removal

4. Marker Gene Search
   - RdRp identification
   <!-- - Classification -->

5. Genome Annotation
   - RNA structure prediction and annotation
   - Protein domain and ORF prediction.

6. Virus Characteristics
   - Feature analysis
   - Report generation

## Output Structure

```
output_dir/
├── filtered_reads/
├── assembly/
│   └── final_assembly.fasta
├── marker_search_results/
│   └── <DB_NAME>_marker_search_results.tsv (raw marker search pyhmmer output).
├── viral_bins/ (note! binning is not yet implemented - each contig is reported as a separate bin for now.)
│   └── bins_summary.tsv
│   └── bin_<<###>>
│       └── bin_<<###>>.fasta
│       └── bin_<<###>>_characteristics.tsv
│       └── genome_annotation/
│           └── bin_<<###>>_genes.tsv \ .gff3 (user-specified format - only ORFs are reported)
│           └── bin_<<###>>_proteins.faa (amino acid sequences)
│           └── bin_<<###>>_genome_annotation.tsv \ .gff3 (user-specified format, RNA and protein annotations)
│           └── bin_<<###>>_ribozymes.tsv 
└── RolyPoly_end_to_end_report.tsv (pipeline summary, concatenated log files, version information, and used tools/dbs citations).
```

## Citations

### RolyPoly
- **RolyPoly**: RNA virus identification pipeline
  - Citation: Neri, U., Bushnell, B., Roux, S., & Camargo, A. P., Steindorff, A.S., Coclet, C., (2024). RolyPoly: A pipeline for RNA virus identification from RNA-seq data. [Software]. Available from https://code.jgi.doe.gov/UNeri/rolypoly

### Read Processing
- **BBMap**: Read processing
  - Citation: https://sourceforge.net/projects/bbmap/files/BBMap_39.08.tar.gz
- **SeqKit**: Sequence manipulation
  - Citation: https://doi.org/10.1002/imt2.191
- **NCBI datasets**: Data retrieval
  - Citation: https://ftp.ncbi.nlm.nih.gov/pub/datasets/command-line/v2/linux-amd64/datasets

### Assembly
- **SPAdes**: Genome assembly
  - Citation: https://doi.org/10.1089/cmb.2012.0021
- **MEGAHIT**: Metagenome assembly
  - Citation: https://doi.org/10.1093/bioinformatics/btv033
- **Penguin**: Virus-aware assembly
  - Citation: https://doi.org/10.1101/2024.03.29.587318
- **Bowtie**: Read mapping
  - Citation: https://doi.org/10.1186/gb-2009-10-3-r25

### Marker Gene Search
- **pyhmmer**: HMM searches
  - Citation: https://doi.org/10.1093/bioinformatics/btad214
- **pyrodigal**: ORF prediction
  - Citation: https://doi.org/10.21105/joss.04296

### Filtering
- **MMseqs2**: Sequence searching
  - Citation: https://doi.org/10.1038/nbt.3988
- **DIAMOND**: Protein searching
  - Citation: https://doi.org/10.1038/nmeth.3176
- **pyfastx**: FASTA handling
  - Citation: https://doi.org/10.1093/bib/bbaa368

### Databases
- **SILVA**: rRNA decontamination
  - Citation: https://doi.org/10.1093/nar/gks1219
- **RefSeq**: DNA filtering
  - Citation: https://doi.org/10.1093%2Fnar%2Fgkv1189
- **RdRp Databases**:
  - NeoRdRp_v2.1: https://doi.org/10.1264/jsme2.ME22001
  - RdRp-scan: https://doi.org/10.1093/ve/veac082
  - RVMT: https://doi.org/10.1016/j.cell.2022.08.023
  - TSA_2018: https://doi.org/10.1093/molbev/msad060
  - Pfam_A_37: https://doi.org/10.1093/nar/gkaa913