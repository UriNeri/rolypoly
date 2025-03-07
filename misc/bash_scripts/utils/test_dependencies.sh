#!/usr/bin/bash

# Written by Uri Neri
# Last modified 13.06.2024 ---- WIP

# ##### Dependency cheakcer #####
Dependency_List=(awk rush seqkit bbmap.sh datasets mmseqs spades.py aws bgzip)
check_dependencies "${Dependency_List[@]}"