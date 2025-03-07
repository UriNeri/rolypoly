# RolyPoly Documentation

!!! warning "🚧 Experimental"
    RolyPoly is under active development - features may be incomplete or experimental.

RolyPoly is an RNA virus analysis toolkit, including a variety of commands and wrappers for external tools (from raw read processing to genome annotation, and back again). It also includes an "end-2-end" command that employs an entire pipeline.

```mermaid
flowchart TD
%% Nodes
A("User input <br> raw reads - fastq files <br> & <br> host genome/exome - fasta file"):::green
B("Read Filtering"):::pink
C{"Assembly"}:::yellow
D("Assembly Filtering"):::pink
E("RdRp pyhmmer Search <br> & <br> Nucleotide search vs known viruses "):::blue
F("Annotation <br> RNA - ribozymes, IRES <br> & <br> Proteins - ORFs/CDS  and functional domains"):::purple
G("Host Predict"):::green
H("Virustyping <br> prediction attempt of life-style, polarity, segment number, capsid type, etc"):::orange

%% Edges
A --> B --> C --> D
D --> E --> F --> G
D --> H
G --> H
H -. Use the assembled, identified viruses for mapping from raw/filtered data.-> E
%% Styling
classDef green fill:#B2DFDB,stroke:#00897B,stroke-width:2px;
classDef orange fill:#FFE0B2,stroke:#FB8C00,stroke-width:2px;
classDef blue fill:#BBDEFB,stroke:#1976D2,stroke-width:2px;
classDef yellow fill:#FFF9C4,stroke:#FBC02D,stroke-width:2px;
classDef pink fill:#F8BBD0,stroke:#C2185B,stroke-width:2px;
classDef purple fill:#E1BEE7,stroke:#8E24AA,stroke-width:2px;
```
 Note: host prediction and virustyping are not implemented yet.


<!-- 
## Table of Contents

1. [Installation](installation.md)
2. [FAQ](FAQ.md)
3. [Workflow Description](workflow.md)
4. [Commands](commands/index.md)
5. [Examples](examples.md)
6. [Tips and Tricks](tips_and_tricks.md)
7. [Configuration](configuration.md)
8. [Contribute](contribute.md)
<!-- 9. [Dependencies](dependencies.md) -->
<!-- 9. [About](about.md)
10. [Citation](citation.md)
11. [Resource Usage](resource_usage.md) -->
<!-- 
 --> 
