# Resource Usage and Requirements
The performance and memory requirements of RolyPoly vary between different commands and types of data. Different steps in the pipeline have varying memory requirements and CPU utilization efficiencies.

We used GNU's `time` command to track resource usage when running different RolyPoly commands on various types of data with different sizes and complexities, ranging from low-depth single isolate mock communities to high-depth metagenomic samples sequenced multiple times.

## Peak RAM Usage

Below is a table showing the peak RAM usage for different RolyPoly commands on various types of data:

| Command | Mock Community (5GB) | Metatranscriptome (20GB) | Large Metagenome (100GB) |
|---------|----------------------|--------------------------|--------------------------|
| prepare_external_data | 8GB | 8GB | 8GB |
| filter_reads | 12GB | 25GB | 60GB |
| assemble | 20GB | 40GB | 120GB |
| filter_assembly | 6GB | 10GB | 25GB |
| annotate | 15GB | 30GB | 70GB |

Note: These values are approximate and may vary depending on the specific dataset and system configuration.

## CPU Utilization

Different commands utilize multiple CPUs with varying efficiencies:

- `prepare_external_data`: Moderate parallelization, benefits from 4-8 cores
- `filter_reads`: Highly parallelizable, scales well up to 16-32 cores
- `assemble`: Varies by assembler, generally benefits from 8-16 cores
- `filter_assembly`: Moderate parallelization, 4-8 cores recommended
- `annotate`: Highly parallelizable, scales well with increasing core count

## Disk Usage

Temporary disk usage can be significant, especially for larger datasets:

- `filter_reads`: 2-5x input size
- `assemble`: 3-10x input size
- `annotate`: 1-3x assembly size

## Runtime

Runtime varies greatly depending on input size and complexity. As a rough guide:

- Small datasets (5-10GB): 2-6 hours
- Medium datasets (20-50GB): 6-24 hours
- Large datasets (100GB+): 1-5 days

<!-- 
## Tips for Optimizing Resource Usage

1. Use SSD storage for temporary files to improve I/O performance
2. Adjust thread count based on your system's capabilities
3. Consider splitting large datasets into smaller chunks for parallel processing
4. Monitor resource usage and adjust parameters as needed -->

Remember that your experience may vary depending on the specific characteristics of your data and computing environment.
