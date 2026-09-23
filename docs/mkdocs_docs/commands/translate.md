# Translate

<!-- Auto-generated draft from CLI metadata for `rolypoly translate`. -->

Translate nucleotide FASTA with RolyPoly's native, IUPAC-aware NumPy backend.
The default mode writes all six end-to-end reading frames and retains the
SeqKit-compatible header convention used by the underlying Python API.

```text
rolypoly translate -i input.fna -o translations.faa
```

Use `--header-format canonical` for RolyPoly IDs such as
`contig_frame_p1`, or use `--defline-template` to control the complete FASTA
defline. Six-frame templates support `id`, `canonical_id`, `description`,
`frame`, `frame_abs`, `strand`, and `frame_token`.

## ORF mode

`--mode orfs` reports start/stop-delimited regions. By default it selects the
first recognized start in each stop-delimited region, includes complete
start-to-stop ORFs, 5′ edge-to-stop partials, and 3′ start-to-edge partials,
and excludes edge-to-edge fragments with neither boundary. Use
`--complete-orfs-only` to exclude partials and `--all-starts` to report nested
ORFs beginning at every recognized start.

Alternative starts come from the selected genetic code and translate as an
initiating methionine. `--atg-only` restricts starts to ATG. ORF templates also
support `orf`, `start`, `end`, and `partial` fields.

```text
rolypoly translate \
  -i input.fna \
  -o orfs.faa \
  --mode orfs \
  --genetic-code 11 \
  --header-format canonical \
  --fna-output orfs.fna \
  --gff-output orfs.gff3 \
  --gff-include-fasta
```

FNA records use coding orientation, including reverse-complement sequence for
minus-strand features. GFF3 coordinates always refer to the original input
sequence. `--gff-include-fasta` appends the original reference records after a
`##FASTA` directive.

<!-- BEGIN GENERATED CLI OPTIONS -->
## Options

- `-i`, `--input`: Input nucleotide FASTA file. (type: `FILE`; required; default: `Sentinel.UNSET`)
- `-o`, `--output`: Output protein FASTA file. (type: `FILE`; required; default: `Sentinel.UNSET`)
- `--mode`: Emit complete six-frame translations or start/stop-delimited ORFs. (type: `CHOICE`; default: `six-frame`)
- `--min-length`: Minimum amino-acid length; ORF-mode terminal stops are excluded. (type: `INTEGER RANGE`; default: `0`)
- `--genetic-code`: NCBI or bundled generic genetic code number. (type: `CHOICE`; default: `1`)
- `--stops-as-x`, `--stops-as-star`: Write resolved stop codons as X or * in protein output. (type: `BOOLEAN`; default: `True`)
- `--header-format`: Use SeqKit-compatible or canonical RolyPoly FASTA identifiers. (type: `CHOICE`; default: `seqkit`)
- `--defline-template`: Custom Python format template overriding --header-format. (type: `TEXT`)
- `--alternative-starts`, `--atg-only`: In ORF mode, recognize the selected genetic code's alternative starts. (type: `BOOLEAN`; default: `True`)
- `--partial-orfs`, `--complete-orfs-only`: In ORF mode, include edge-to-stop and start-to-edge partial ORFs. (type: `BOOLEAN`; default: `True`)
- `--all-starts`, `--longest-only`: In ORF mode, emit every nested start or only the first start per region. (type: `BOOLEAN`; default: `False`)
- `--fna-output`: Optionally write translated nucleotide regions in coding orientation. (type: `FILE`)
- `--gff-output`: Optionally write translated regions or ORFs as GFF3. (type: `FILE`)
- `--gff-include-fasta`: Append the input reference FASTA to --gff-output. (type: `BOOLEAN`; default: `False`)
- `-t`, `--threads`: Number of worker threads. (type: `INTEGER RANGE`; default: `1`)
- `-g`, `--log-file`: Path to the log file. (type: `FILE`; default: `rolypoly.log`)
<!-- END GENERATED CLI OPTIONS -->

## Caveats

ORF mode reports sequence-intrinsic start/stop regions; **it is NOT a statistical gene predictor** and can produce many incidental ORFs in long or metagenomic
sequences. Partial ORFs describe truncation at the supplied contig boundary,
which does not by itself establish biological truncation.
