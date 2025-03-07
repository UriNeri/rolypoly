# FAX Module
## API Reference

::: rolypoly.utils.fax
    options:
        show_root_heading: false
        show_source: false

<!-- 
## Usage Examples

### Reading FASTA Files

```python
from rolypoly.utils.fax import read_fasta_needletail, read_fasta_polars

# Using needletail (fast, memory-efficient)
seq_ids, seqs = read_fasta_needletail("input.fasta")

# Using polars (returns DataFrame)
df = read_fasta_polars("input.fasta", idcol="seq_id", seqcol="sequence")
```

### Sequence Translation and ORF Prediction

```python
from rolypoly.utils.fax import translate_6frx_seqkit, pyro_predict_orfs

# 6-frame translation
translate_6frx_seqkit("input.fasta", "translated.faa", threads=4)

# ORF prediction with pyrodigal-gv
pyro_predict_orfs("input.fasta", "orfs.faa", threads=4)
```

### HMM Database Operations

```python
from rolypoly.utils.fax import search_hmmdb, hmm_from_msa

# Search protein sequences against HMM database
search_hmmdb(
    "proteins.faa", 
    "pfam.hmm", 
    "results.txt",
    threads=4,
    match_region=True,
    full_qseq=True
)

# Create HMM from multiple sequence alignment
hmm_from_msa("alignment.fasta", "output.hmm", alphabet="amino")
```  -->