# Examples

This page provides minimal working command examples using the current CLI.

## Read Processing

```bash
rolypoly filter-reads -i reads_R1.fq,reads_R2.fq -o filtered_reads/
```

## Assembly

```bash
rolypoly assemble -id filtered_reads/ -o assembly_out/
```

## Marker Search

```bash
rolypoly marker-search -i assembly_out/final_assembly.fasta -o marker_out/
```

## Virus Mapping

```bash
rolypoly virus-mapping -i assembly_out/final_assembly.fasta -o virus_hits.tab --db all
```

## End-to-End

```bash
rolypoly end2end -i reads_R1.fq,reads_R2.fq -o rp_e2e_out/
```
