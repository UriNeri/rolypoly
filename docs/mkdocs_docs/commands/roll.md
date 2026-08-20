# Roll

<!-- Auto-generated draft from CLI metadata for `rolypoly roll`. -->
<!-- Expand this page with command-specific context, examples, and citations. -->

## Summary

End-to-end pipeline for RNA virus discovery from raw sequencing data.

## Description

This pipeline performs a complete analysis workflow including:
1. Read filtering and quality control (optionally, subsampling too)
2. De novo assembly
3. Contig filtering
4. Marker gene search (default: RdRps + genomad) and nucleic search (default: known RNA viruses)
5. Genome annotation (default: NVPC + Pfam for proteins, and Rfam+linerfold for catalytic/structural RNAs)
6. Optional ICTV taxonomy assignment with mmtax
7. Virus characteristics prediction - NOT IMPLEMENTED YET

## Usage

```bash
rolypoly roll [OPTIONS]
```

## Options

- `-i`, `--input`: Input path to raw RNA-seq data (fastq/gz file or directory with fastq/gz files) (type: `TEXT`; required; default: `Sentinel.UNSET`)
- `-o`, `--output-dir`: Output directory (type: `TEXT`; default: `/home/neri/Documents/GitHub/rps/rolypoly_rp_e2e`)
- `-D`, `--host`: Path to the user-supplied host/contamination fasta /// Fasta file of known DNA entities expected in the sample. If not provided some steps will be skippted. (type: `TEXT`)
- `--preset`: Preset that selects both a filter-reads and an assemble preset suited to the library type. 'rna_virus': RNA virus metatranscriptome (default): rRNA removal (mincovfraction=0.6), host + identified-DNA filtering, no polyA trim; rnaviralSPAdes+MEGAHIT assembly 'ribodepleted': Total-RNA ribo-depleted: stricter rRNA removal (mincovfraction=0.65), host + identified-DNA filtering, no polyA trim; rnaviralSPAdes+MEGAHIT assembly 'poly_a': Poly-A selected mRNA: polyA tail trimming (trimpolya=18), stricter quality trim (trimq=12); rnaSPAdes+MEGAHIT assembly 'all_virus_metat': All-virus metatranscriptome / RNA virome: relaxed rRNA removal (mincovfraction=0.5), skips identified-DNA filter; rnaviralSPAdes+MEGAHIT assembly 'DNA_virus': DNA virome / metagenomics: skips rRNA and identified-DNA filtering entirely; metaSPAdes only 'complete': Expansive: rna_virus_metat read filtering + all three assemblers (metaSPAdes+rnaviralSPAdes+MEGAHIT) 'fast': Quick preview / mini mode: subsamples reads, skips error correction and identified-DNA filter; MEGAHIT only with narrow k-mer range; narrowed marker/nucleic/annotation databases (type: `CHOICE`; default: `rna_virus`)
- `--filter-preset`: Override the read-filtering preset chosen by --preset. (type: `CHOICE`)
- `--assembly-preset`: Override the assembly preset chosen by --preset. (type: `CHOICE`)
- `--mini`: Enable mini mode for quick testing. This will subsample the input reads and use a faster assembly preset. (type: `BOOLEAN`; default: `False`)
- `-sz`, `--sample-size`: Total reads (>1) OR proportion (0-1) of total reads to be used by --mini subsampling. NOTE: this is ignored if --mini-subset-type is set to bbnorm (type: `INTEGER`; default: `50000`)
- `-ml`, `--min-len`, `--minimum-length`: Contigs shorter than this will not be used during virus identification (i.e. in marker search OR nucleic search) (type: `INTEGER`; default: `200`)
- `-mst`, `--mini-subset-type`: Subset type used if --mini is set. note: first is quicker than random which is quicker than bbnorm, but bbnorm is the only one that might be useful in a non 'quick and dirty' attempt. 'first' assumes your input isn't sorted by anything. (type: `CHOICE`; default: `random`)
- `--skip-existing`: Skip commands if output files already exist (type: `BOOLEAN`; default: `False`)
- `-ow`, `--overwrite`: Overwrite roll output directory if it already exists (type: `BOOLEAN`; default: `False`)
- `-A`, `--assembler`: Assembler choice (spades,megahit,penguin). For multiple, give a comma-separated list (type: `TEXT`; default: `spades,megahit`)
- `--no-rmdup`: Disable default assembly dereplication before downstream analysis. (type: `BOOLEAN`; default: `False`)
- `-m`, `--mapper`: Mapper backend(s) for standalone read mapping. Use multiple -m flags for multiple mappers. select 'none' to skip mapping. (type: `CHOICE`; default: `bbmap`)
- `--cluster-backend`: ANI backend for contig clustering. Use 'none' to skip clustering entirely. (type: `CHOICE`; default: `linclust`)
- `--cluster-method`: Clustering method passed to the cluster command. (type: `CHOICE`; default: `centroid`)
- `--cluster-min-identity`: Minimum identity threshold for clustering. (type: `FLOAT`; default: `99.0`)
- `--cluster-min-target-coverage`: Minimum target coverage threshold for clustering. (type: `FLOAT`; default: `99.0`)
- `--cluster-min-query-coverage`: Minimum query coverage threshold for clustering. (type: `FLOAT`; default: `0.0`)
- `--cluster-min-alignment-fraction`: Minimum min(query,target) coverage threshold for clustering. (type: `FLOAT`; default: `0.0`)
- `--cluster-mmseqs-sensitivity`: MMseqs sensitivity when cluster backend uses mmseqs. (type: `FLOAT`; default: `7.5`)
- `-Fm1`, `--filter1_nuc`: First set of rules for nucleic filtering by aligned stats (type: `TEXT`; default: `alnlen >= 120 & pident>=75`)
- `-Fm2`, `--filter2_nuc`: Second set of rules for nucleic match filtering (type: `TEXT`; default: `qcov >= 0.95 & pident>=95`)
- `-Fd1`, `--filter1_aa`: First set of rules for amino (protein) match filtering (type: `TEXT`; default: `length >= 80 & pident>=75`)
- `-Fd2`, `--filter2_aa`: Second set of rules for protein match filtering (out potential host/contamination sequences) (type: `TEXT`; default: `qcovhsp >= 95 & pident>=80`)
- `--dont-mask`: If set, host fasta won't be masked for potential RNA virus-like seqs (type: `BOOLEAN`; default: `False`)
- `--mmseqs-args`: Additional arguments to pass to MMseqs2 search command during filtering of potential host/contamination sequences (type: `TEXT`; default: `--min-seq-id 0.5 --min-aln-len 80`)
- `--diamond-args`: Additional arguments to pass to Diamond search command during filtering of potential host/contamination sequences (type: `TEXT`; default: `--id 50 --min-orf 50`)
- `--dbm`, `--db-marker`: Database(s) to use for marker gene search (type: `TEXT`; default: `rvmt,genomad,Pfam_RTs_RdRp`)
- `--dbn`, `--db-nucleic`: Database(s) to use for nucleic acid search (type: `TEXT`; default: `all`)
- `--dba`, `--db-annotation`: Database(s) to use for protein annotation. (type: `TEXT`; default: `all`)
- `-txb`, `--taxonomy-backend`: No description provided. (type: `CHOICE`; default: `mmseqs`)
- `-txd`, `--taxonomy-db`: Built-in mmtax database name or custom backend database path. (type: `TEXT`; default: `ncbi_virus`)
- `-txt`, `--taxonomy-taxdump`: Taxdump required when --taxonomy-db is a custom path. (type: `DIRECTORY`)
- `-txs`, `--taxonomy-sensitivity`: Shared mmtax sensitivity preset or level 1-8. (type: `TEXT`; default: `normal`)
- `--report`, `--no-report`: Write an interactive HTML genome-map report (genome_maps.html) from the annotation results (marker/protein hits + RNA track) at the end of the run. (type: `BOOLEAN`; default: `True`)
- `--report-best-by`: Initial 'best hit per range' criterion shown in the report (toggleable in the viewer). (type: `CHOICE`; default: `score`)
- `-t`, `--threads`: Number of worker threads. (type: `INTEGER RANGE`; default: `1`)
- `-M`, `--memory`: Memory limit, for example 8g. (type: `MEMORY`; default: `8g`)
- `-k`, `--keep-tmp`: Keep temporary files. (type: `BOOLEAN`; default: `False`)
- `-tmp`, `--temp-dir`: Temporary working directory. (type: `DIRECTORY`)
- `-g`, `--log-file`: Path to the log file. (type: `FILE`; default: `rolypoly.log`)




