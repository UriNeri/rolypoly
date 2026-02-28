# Assembly Filtering

`filter-contigs` removes potential host and contamination sequences through nucleotide and protein-level comparisons.

```mermaid
flowchart TD
	subgraph IN["<b>Input</b>"]
		IN1["Contigs FASTA (.fa/.fasta)"]
		HOST["Host/contamination FASTA (.fa/.fasta)"]
	end

	subgraph P["<b>Filtering Workflow</b>"]
		MASK["Mask Host (optional)<br> (bbmap/minimap2)"]
		NUC["Nucleotide Filter<br> (mmseqs2 -> rules: alnlen,pident,qcov)"]
		AA["Protein Filter<br> (diamond -> rules: length,pident,qcovhsp)"]
		MERGE["Merge & Finalize<br> (filtered_contigs.fasta)"]
	end

	subgraph OUT["<b>Outputs</b>"]
		OUTF["filtered_contigs.fasta"]
		LOG["filter_contigs_log.txt"]
	end

	IN1 --> NUC
	HOST --> MASK --> NUC
	NUC --> AA
	AA --> MERGE --> OUTF
	MERGE --> LOG

	classDef inputStyle fill:#f0f9ff,stroke:#0366d6,color:#03396c;
	classDef pipelineStyle fill:#fffaf0,stroke:#b85c00,color:#7a3b00;
	classDef outputStyle fill:#f0fff4,stroke:#0b8a3e,color:#0b6624;

	class IN1,HOST inputStyle
	class MASK,NUC,AA,MERGE pipelineStyle
	class OUTF,LOG outputStyle
```

## Options

### Common
- `-i, --input`: Input fasta file (required)
- `-o, --output`: Output file location (default: current_directory/filtered_contigs.fasta)
- `-t, --threads`: Number of threads (default: 1)
- `-M, --memory`: Memory allocation (default: "6g")
- `-g, --log-file`: Path to log file
- `--keep-tmp`: Keep temporary files (flag)

### Filtering
- `-m, --mode`: Operation mode: nuc, aa, or both (default: 'both')
- `-d, --known-dna, --host`: Host/contamination fasta file (required)
- `--dont-mask`: Skip masking RNA virus-like sequences in host fasta (flag)

### Nucleotide Filtering
- `-Fm1, --filter1_nuc`: First filter (default: "alnlen >= 120 & pident>=75")
- `-Fm2, --filter2_nuc`: Second filter (default: "qcov >= 0.95 & pident>=95")
- `--mmseqs-args`: Additional MMseqs2 arguments (default: "--min-seq-id 0.5 --min-aln-len 80")

### Protein Filtering
- `-Fd1, --filter1_aa`: First filter (default: "length >= 80 & pident>=75")
- `-Fd2, --filter2_aa`: Second filter (default: "qcovhsp >= 95 & pident>=80")
- `--diamond-args`: Additional Diamond arguments (default: "--id 50 --min-orf 50")

### Filter Variables
- Nucleotide: `alnlen`, `pident`, `qcov`
- Protein: `length`, `pident`, `qcovhsp`

## Citations

This command uses the following tools:

### Search Tools
- MMseqs2: https://doi.org/10.1038/nbt.3988
- DIAMOND: https://doi.org/10.1038/nmeth.3176

### Support Tools
- BBMap: https://sourceforge.net/projects/bbmap/files/BBMap_39.08.tar.gz
- pyrodigal: https://doi.org/10.21105/joss.04296
- pyfastx: https://doi.org/10.1093/bib/bbaa368 