About RolyPoly
==============
RolyPoly is an RNA virus analysis toolkit, including a variety of commands and wrappers for external tools (from raw read processing to genome annotation, and back again). It also includes an "end-2-end" command that employs an entire pipeline. 


Motivation
----------
Current workflows for RNA virus detection are functional but could be improved, especially by utilizing raw reads instead of pre-existing, general-purpose made, assemblies. Here we proced with more specific processes tailored for RNA viruses.
Currently, these processes are known to people but are not concentrated in a sinflge place, so people either have to figure out how to implement the procedures themselves or scavenge code from multiple repositories in different languages. That said, several similar software exist, but have different uses, for example:  
- hecatomb (github.com/shandley/hecatomb): uses mmseqs for homology detection and thus is less sensitive then the additional HMMer based identification herein.
- AliMarko (biorxiv.org/content/10.1101/2024.07.19.603887): Utilizes a single-sample assembly only approach, not supporting co/cross assembly of multiple samples. Additionally, AliMarko uses a small, partially outdated (IMO) HMM profile set.


Authors
-------

- Uri Neri
- Brian Bushnell
- Simon Roux
- Antônio Pedro Camargo
- Andrei Stecca Steindorff
- Clement Coclet
- David Parker
- ...
 

Contact
---------------
TBD


Acknowledgments
---------------

Thanks to the DOE Joint Genome Institute for infrastructure support.
Special thanks to all contributors who have offered insights and
improvements.

