## Tips and Tricks
- Each of the main commands of rolypoly could be entered into using external inputs (e.g. you already have assembly and want to analyse it).
- If you have a lot of similar samples, some operations might be preformed once instead of rerunning the end-2-end command. For example, if you are working on the same host (or if ytou suspect the DNA cotanaminats in your samples to be consistent across multiple runs) you can mask the host genome once, externally, provide it to rolypoly's mask_dna, and then when running the `filter_*` commands, use the flag "dont_mask" to skip masking. 
- Offloading commands to different machines is a smart idea if your access to a `bigmem` compute node is not a given. 
