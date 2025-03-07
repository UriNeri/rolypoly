



 #### WIP: ####
# import biobear as bb
# import genomicranges as gg

# test = bb.FastaReader(("./merged_contigs.fasta")).to_polars()
# test_mmseqs_out = pl.read_csv(f'merged_contigs_clst_rep_seq_vs_host.tab',separator="\t",has_header=True) #,new_columns= "qheader,theader,qlen,tlen,qstart,qend,tstart,tend,alnlen,mismatch,qcov,tcov,bits,evalue,gapopen,pident,nident".split(",")
# # Add a new column for strand
# test_mmseqs_out = test_mmseqs_out.with_columns(
#     pl.when(pl.col("qstart") > pl.col("qend"))
#       .then(pl.lit("+"))
#       .otherwise(pl.lit("-"))
#       .alias("strand"))

# test_mmseqs_out = test_mmseqs_out.with_columns(
#     pl.when(pl.col("strand") == "-")
#       .then(pl.col("qend"))
#       .otherwise(pl.col("qstart"))
#       .alias("adjusted_qstart"))

# test_mmseqs_out = test_mmseqs_out.with_columns(
#     pl.when(pl.col("strand") == "-")
#       .then(pl.col("qstart"))
#       .otherwise(pl.col("qend"))
#       .alias("adjusted_qend"))

# test_mmseqs_out["qend","qstart","adjusted_qend","adjusted_qstart","strand"]

# test_mmseqs_out = test_mmseqs_out.with_columns([
#     pl.col("adjusted_qstart").alias("qstart"),
#     pl.col("adjusted_qend").alias("qend"),
# ])

# # Drop temporary columns
# test_mmseqs_out = test_mmseqs_out.drop(["adjusted_qstart", "adjusted_qend"])



# test_mmseqs_out = test_mmseqs_out.rename({"qheader": "seqnames", "qstart": "starts", "qend": "ends"})
# test_mmseqs_out = test_mmseqs_out["seqnames","ends","starts","strand"].unique().sort(by=["seqnames","strand"])

# # df = session.read_gtf_file("path/to/test.gtf").to_polars()

# asd = gg.GenomicRanges.from_polars(test_mmseqs_out)


 #### OBSOLETE: ####
# def get_resource_usage():
#     process = psutil.Process(os.getpid())
#     cpu_percent = process.cpu_percent(interval=1)
#     memory_info = process.memory_info()
#     return f"CPU: {cpu_percent:.1f}% | Memory: {memory_info.rss / (1024 * 1024):.1f} MB"


# def logit(logger, message, include_resources=False):
#     if include_resources:
#         # resource_info = get_resource_usage()
#         message = f"{message} | {resource_info}"
#     logger.info(message)

