# import os as os
# import subprocess
# import pathlib as Path
# from rolypoly.utils.various import fetch_and_extract
# from rolypoly.utils.fax import write_fastx_file 
# from needletail import parse_fastx_file
# def create_directory(dir_name):
#     if not os.path.exists(dir_name):
#         os.makedirs(dir_name)

# def write_fasta(records, filename):
#     with open(filename, "w") as output_handle:
#         write_fastx_file(records, output_handle, "fasta")

# def write_gbff(records, filename):
#     with open(filename, "w") as output_handle:
#         write_fastx_file(records, output_handle, "genbank")

# def dissect_big_gbff(gbff_file):
#     # Dictionary to hold records categorized by translation table
#     records_by_table = {}

#     # Parse the GBFF file
#     for record in parse_fastx_file(gbff_file, "genbank"):
#         if "complete genome" not in record.description:
#             continue

#         table = None
#         genome_seq = record.seq
        
#         # Get the translation table from the features
#         for feature in record.features:
#             if feature.type == "CDS" and "transl_table" in feature.qualifiers:
#                 table = feature.qualifiers["transl_table"][0]
#                 break
        
#         if table is None:
#             table = "unknown"
        
#         if table not in records_by_table:
#             records_by_table[table] = {
#                 "genome": [],
#                 "genes": [],
#                 "proteins": [],
#                 "gbff_records": []
#             }
        
#         records_by_table[table]["genome"].append(SeqRecord(genome_seq, id=record.id, description=record.description))
#         records_by_table[table]["gbff_records"].append(record)
        
#         # Extract gene and protein sequences
#         for feature in record.features:
#             if feature.type == "CDS":
#                 gene_seq = feature.extract(record.seq)
#                 protein_seq = feature.qualifiers.get("translation", [""])[0]
                
#                 gene_record = SeqRecord(gene_seq, id=feature.qualifiers.get("locus_tag", [""])[0], description="")
#                 protein_record = SeqRecord(protein_seq, id=feature.qualifiers.get("locus_tag", [""])[0], description="")
                
#                 records_by_table[table]["genes"].append(gene_record)
#                 records_by_table[table]["proteins"].append(protein_record)

#     # Create directories and write to files
#     for table, data in records_by_table.items():
#         dir_name = f"table_{table}"
#         create_directory(dir_name)
        
#         # Write genome sequences
#         genome_file = os.path.join(dir_name, "genome.fasta")
#         write_fasta(data["genome"], genome_file)
        
#         # Write gene sequences
#         gene_file = os.path.join(dir_name, "gene.fna")
#         write_fasta(data["genes"], gene_file)
        
#         # Write protein sequences
#         protein_file = os.path.join(dir_name, "protein.faa")
#         write_fasta(data["proteins"], protein_file)
        
#         # Write GBFF file
#         gbff_file = os.path.join(dir_name, "records.gbff")
#         write_gbff(data["gbff_records"], gbff_file)


# if __name__ == "__main__":
#     os.chdir("/clusterfs/jgi/scratch/science/metagen/neri/rolypoly/data/NCBI_ribovirus/test_progv")
#     fetch_and_extract(url="https://ftp.ncbi.nlm.nih.gov/refseq/release/viral/viral.1.genomic.gbff.gz",extract_to="viral.1.genomic.gbff")

#     gbff_file = "viral.1.genomic.gbff"  
#     dissect_big_gbff(gbff_file)
#     big_command="""
# # git clone -b print-model https://github.com/apcamargo/prodigal-gv.git;
# # cd prodigal-gv;
# make;
# export prodigal-train={pwd}/prodigal-gv
#     """
#     subprocess.run(big_command,shell=True)
#     prodigal_train = os.path.abspath(path="prodigal-gv")
#     os.chdir("/clusterfs/jgi/scratch/science/metagen/neri/rolypoly/data/NCBI_ribovirus/test_progv")

#     subprocess.run(f"wget https://raw.githubusercontent.com/UriNeri/ColabScan/main/selected_gencode_6ers.fasta ; cat selected_gencode_6ers.fasta >> table_6/genome.fasta",shell=True )
    
#     for table in list(Path.Path('.').glob('table_*')):
#         temp_t=str(table).replace("table_","")
#         if(temp_t!="unknown"):
#             prodigal_train_cmd=f"{prodigal_train} -t {str(table)}_vrfsq_model.trn -i {str(table)}/genome.fasta -g {temp_t} -n  > model_{str(table)}_vrfsq_model.trn"
#             subprocess.run(prodigal_train_cmd,shell=True)




