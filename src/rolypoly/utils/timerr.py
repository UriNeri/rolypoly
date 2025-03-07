# from ctypes import sizeof
# import os
# import re
# import glob
# import subprocess
# import polars as pl
# from pathlib import Path
# import datetime
# import json


# # Clean up column names
# def convert_mem_mb(col):
#     """ get's a string with a number and a letter K,M,G and returns the number * 1024 if the letter is K, * 1024*1024 if the letter is M, * 1024*1024*1024 if the letter is G"""
#     if col is None:
#         return None
#     if col.endswith("K"):
#         return float(col[:-1]) * 1024
#     elif col.endswith("M"):
#         return float(col[:-1]) * 1024 * 1024
#     elif col.endswith("G"):
#         return float(col[:-1]) * 1024 * 1024 * 1024
#     else:
#         return float(col)
    



# file_path = "/REDACTED_HPC_PATH/projects/myco/slurmsout/sacct.out"
# def read_sacct_data(file_path):
#     sacct_df = pl.read_csv(file_path, separator='|',truncate_ragged_lines=True)
#     tmp_df = pl.read_csv("/REDACTED_HPC_PATH/projects/myco/slurmsout/filter_reads/run_id_sample_id.csv")
#     sacct_df = sacct_df.join(tmp_df,on="JobID",how="left")
#     # remove jobs whose runtime exceeds 24 hours
#     faulty_elapsedtime= sacct_df.filter(pl.col('Elapsed').str.extract('(-)').is_not_null()).select(pl.col('JobID')).to_series().to_list()
#     faulty_elapsedtime += sacct_df.filter(pl.col('State').is_in(["FAILED", "CANCELLED"])).select(pl.col('JobID')).to_series().to_list()
    
#     # faulty_elapsedtime += sacct_df.filter(pl.col('run_id').is_null()).select(pl.col('JobID')).to_series().to_list()
#     sacct_df = sacct_df.filter(~pl.col('JobID').is_in(faulty_elapsedtime))
#     # Extract and clean up additional columns
#     sacct_df = sacct_df.with_columns([
#         pl.col('JobID').str.split('_').list.first().alias('arrayID'),
#         pl.col('JobID').str.split('_').list.last().str.split(".").list.last().alias('slurm_step'),
#         pl.col('JobID').str.split('_').list.last().str.split(".").list.first().alias('JobID_in_array'),
#         pl.col('Elapsed'),
#         pl.col('State'),
#         pl.col('ExitCode').str.split(":").list.last().cast(pl.UInt16).alias("ExitCode"),
#         pl.col('AllocCPUS').cast(pl.UInt16),
#         pl.col("run_id").cast(pl.Utf8)
#     ])
    
#     sacct_df = sacct_df.with_columns(
#         pl.col('MaxVMSize').map_elements(convert_mem_mb, return_dtype=pl.Float64).alias("MaxVMSize_B"),
#         pl.col('MaxRSS').map_elements(convert_mem_mb, return_dtype=pl.Float64).alias("MaxRSS_B"),
#         pl.col('ReqMem').map_elements(convert_mem_mb, return_dtype=pl.Float64).alias("ReqMem_B"),
#         pl.col('MaxDiskRead').map_elements(convert_mem_mb, return_dtype=pl.Float64).alias("MaxDiskRead_B"),
#         pl.col('MaxDiskWrite').map_elements(convert_mem_mb, return_dtype=pl.Float64).alias("MaxDiskWrite_B"),
#     )
    
#     # Convert Elapsed to seconds
#     sacct_df = sacct_df.with_columns(
#         pl.col('Elapsed').str.split(':').map_elements(
#             lambda x: int(x[0].split(r'[^\d]')[0])*3600 + int(x[1].split(r'[^\d]')[0])*60 + int(x[2].split(r'[^\d]')[0]),
#             return_dtype=pl.Int64
#         ).alias('elapsed_seconds')
#     )
#     # sacct_df = sacct_df.select([
#     #     pl.col("run_id"),
#     #     pl.col("arrayID"),
#     #     pl.col("JobID"),
#     #     pl.col("elapsed_seconds"),
#     #     pl.col("MaxRSS_B"),
#     #     pl.col("MaxVMSize_B"),
#     #     pl.col("MaxDiskRead_B"),
#     #     pl.col("MaxDiskWrite_B"),
#     #     pl.col("TRESUsageInTot"),
#     #     pl.col("ExitCode"),
#     #     pl.col("AllocCPUS"),
#     #     pl.col("ReqMem_B"),
#     #     pl.col("State"),
#     #     pl.col("slurm_step")
#     # ])
    
#     # Now, we'll fill missing null values by the max value of the column for each jobid
#     tmpgroup = sacct_df.group_by(pl.col("arrayID").drop_nulls(),pl.col("JobID_in_array").drop_nulls()).agg( 
#         pl.col("MaxRSS_B").drop_nulls().max().alias("MaxRSS_B"),
#         pl.col("MaxVMSize_B").drop_nulls().max().alias("MaxVMSize_B"),
#         pl.col("MaxDiskRead_B").drop_nulls().max().alias("MaxDiskRead_B"),
#         pl.col("MaxDiskWrite_B").drop_nulls().max().alias("MaxDiskWrite_B"),
#         pl.col("TRESUsageInTot").drop_nulls().max().alias("TRESUsageInTot"),
#         pl.col("ReqMem_B").drop_nulls().max().alias("ReqMem_B"),
#         pl.col("ExitCode").drop_nulls().max().alias("ExitCode"),
#         pl.col("elapsed_seconds").drop_nulls().max().alias("elapsed_seconds"),
#         pl.col("AllocCPUS").drop_nulls().max().alias("AllocCPUS"),
#         # pl.col("State").drop_nulls().str.join(" ").str.join(", ").alias("State"),
#         pl.col("slurm_step").drop_nulls().unique().str.join(", ").alias("slurm_step"),
#         pl.col("JobID").drop_nulls().unique().str.join(", ").alias("JobID"),
#         # pl.col("arrayID").drop_nulls().unique().alias("arrayID"),
#     )
#     (tmpgroup).write_csv("tmpgroup.csv") 
#     return sacct_df


# # jobs_arrays_mycocosm={
# # 10565899,
# # 10566039,
# # 10613129,
# # 10613272,
# # 10614038,
# # 10618049,
# # 10649795,
# # 10659296,
# # 10688002,
# # 10688088,
# # 10689715,
# # }


# def log_to_table(log_file):
#     with open(log_file, 'r') as f:
#         content = f.read()
#     time_list=[]
#     level_list=[]
#     message_list=[]
#     for line in content.split("\n"):
#         if line == "":
#             continue
#         if len(line.split(" --- ")) == 3:
#             timedate = line.split(" --- ")[0].strip()
#             level = line.split(" --- ")[1].strip()
#             message = line.split(" --- ")[2].strip()
#             time_list.append(datetime.datetime.strptime(timedate, '%Y-%m-%d %H:%M:%S'))
#             level_list.append(level)
#             message_list.append(message)
#             continue
            
#         sep1=line.split(" - ")
#         if(len(sep1) == 2):
#             timedate = sep1[0]
#             sep2 = sep1[1].split(" : ")
#             level = sep2[0]
#             message = sep2[1].strip()
#             time_list.append(datetime.datetime.strptime(timedate, '%Y-%m-%d %H:%M:%S'))
#             level_list.append(level)
#             message_list.append(message)
#             continue
        
#         elif(len(sep1) == 3):
#             timedate = sep1[0]
#             level = sep1[1]
#             message = sep1[2].strip()
#             time_list.append(datetime.datetime.strptime(timedate, '%Y-%m-%d %H:%M:%S'))
#             level_list.append(level)
#             message_list.append(message)
#             continue
#         else:
#             print(sep1)
#             print(log_file)
#             print(len(line.split(" - ")))


#     df = pl.DataFrame({"time": time_list, "level": level_list, "message": message_list})
#     return df
        
# def time_to_table(time_file):
#     with open(time_file, 'r') as f:
#         content = f.read()
    
#     time_dict = {}
#     for line in content.split("\n"):
#         if line != "":
#             key, value = line.split(": ")
#             time_dict[key.strip()] = value.strip()
#     df = pl.from_dict(time_dict)
#     # df = df.with_columns(pl.lit(time_file.split("/")[-1].split("_")[0]).alias("run_name"))
#     df = df.with_columns(
#         pl.col("Command being timed")
#         .str.extract(r"--threads (\d+)|-t (\d+)")
#         .fill_null(strategy="forward")
#         .cast(pl.Int64)
#         .alias("threads_provided")
#     )
#     return df

# def read_bbstats_files(stats_files):
# #File	/REDACTED_HPC_PATH/projects/myco/reads/spids/1298387/1298387_filtered_reads/1298387_cated.fastq.gz
# #Total	92283118
# #Matched	88950521	96.38873%
#     df = pl.DataFrame()
#     for stats_file in stats_files:
#         with open(stats_file, 'r') as f_in:
#             content = f_in.read()
#         tmp_dict = {}
#         lines = content.split("\n")
#         tmp_dict["input_file_path"] = lines[0].split("\t")[1].strip()
#         tmp_dict["total_reads"] = int(lines[1].split("\t")[1].strip())
#         tmp_dict["matched_reads"] = int(lines[2].split("\t")[1].strip())
#         tmp_dict["matched_prop"] = lines[2].split("\t")[2].strip()
#         tmp_dict["command_name"] = "_".join(stats_file.split("/")[-1].split("_")[1:-2]) # last two _ are the """file_name" or original input's base name.
#         df = df.vstack(pl.from_dict(tmp_dict))
#         df=df.with_columns(pl.col("input_file_path").cast(pl.Utf8),
#                           pl.col("command_name").cast(pl.Utf8),
#                           pl.col("total_reads").cast(pl.Int64),
#                           pl.col("matched_reads").cast(pl.Int64),
#                           pl.col("matched_prop").cast(pl.Utf8))
#     return df


# def process_run_data(run_id):
#     basepath_run_info = f"/REDACTED_HPC_PATH/projects/myco/reads/spids/{str(run_id)}/{str(run_id)}_filtered_reads/run_info/"
#     try:
#         log_file = glob.glob(os.path.join(basepath_run_info, '*_log.txt'))[0]
#         time_file = glob.glob(os.path.join(basepath_run_info, '*.time'))[0]
#         config_file = glob.glob(os.path.join(basepath_run_info, '*.json'))[0]
#         output_tracker_file = glob.glob(os.path.join(basepath_run_info, 'output_tracker.csv'))[0]
#         stats_files = glob.glob(os.path.join(basepath_run_info, 'stats_*.txt'))
#     except:
#         print("some files could not be found, skipping ")
#         return
        
#     if not os.path.exists(log_file):
#         print(f"Missing log file for {run_id}")
#         return None
#     if not os.path.exists(time_file):
#         print(f"Missing time file for {run_id}")
#         return None
#     if not os.path.exists(config_file):
#         print(f"Missing config file for {run_id}")
#         return None
#     if not os.path.exists(output_tracker_file):
#         print(f"Missing output tracker file for {run_id}")
#         return None
#     if len(stats_files) < 3:
#         print(f"Not enough stats files found for {run_id}")
#         return None
    
#     time_data = time_to_table(time_file)
#     rpconfig = json.load(open(config_file))
#     stats_table = read_bbstats_files(stats_files)
    
#     output_tracker_data = pl.read_csv(output_tracker_file)
#     output_tracker_data = output_tracker_data.rename({"filename": "output_filename"})
    
#     test_df = output_tracker_data.join(stats_table, on="command_name", how="left")
    
#     # print(log_file)
#     log_data = log_to_table(log_file)
#     if len(log_data) == 0:
#         return None, None

#     log_data = log_data.with_columns(
#         pl.col("message").str.extract(r'Starting step\: (.*)').alias("start_command_name"),
#         pl.col("message").str.extract(r'Finished step\: (.*)').alias("end_command_name"),
#         pl.col("message").str.extract(r'Skipping step\: (.*)').alias("skip_command_name"),
#     )
#     tmp_dict = {}
#     for command_name in log_data["start_command_name"].drop_nulls().unique():
#         if command_name not in rpconfig["skip_steps"]:
#             tmp_dict[command_name] = log_data.filter(pl.col("end_command_name") == command_name).select(pl.col("time")).item() - log_data.filter(pl.col("start_command_name") == command_name).select(pl.col("time")).item()

#     if len(tmp_dict) == 0:
#         print(f"No data found in {log_file}")
#         return None, None
#     runtime_df = pl.DataFrame({"runtime": list(tmp_dict.values()), "command_name": list(tmp_dict.keys())})

#     # step_order_from_log = log_data.select(pl.col("message").str.extract(r'Starting step\: (.*)').alias("step")).to_series().drop_nulls().to_list() # does not include the final deduping.
   
#     output_tracker_data = test_df.pipe(calc_command_time)
    
#     merged_df = runtime_df.join(output_tracker_data, on="command_name", how="left")
    
#     total_runtime_from_timerr = datetime.datetime.strptime(time_data.select(pl.col("Elapsed (wall clock) time (h:mm:ss or m:ss)").cast(str)).item(), '%M:%S.%f').strftime('%H:%M:%S')
#     time_data = time_data.with_columns(pl.lit(total_runtime_from_timerr).alias("total_runtime_from_timerr"))
#     time_data = time_data.with_columns(pl.lit(stats_table.sort(by="total_reads", descending=True).select(pl.col("input_file_path"))[0].item()).alias("input_file_path"))
#     time_data = time_data.with_columns(pl.lit(stats_table.sort(by="total_reads", descending=True).select(pl.col("total_reads"))[0].item()).alias("total_reads"))
#     time_data = time_data.with_columns(pl.lit("filter_reads").alias("command_name"))
    
#     return time_data, merged_df
    
# def calc_command_time(df): 
#     df = df.with_columns((
#         pl.col("timestamp").cast(pl.Datetime) - pl.col("timestamp").shift(1).cast(pl.Datetime)).alias("runtime")
#     )
#     return df   
    
# def flatten_complex_element(element):
#     if isinstance(element, list):
#         return ", ".join(str(item) for item in element)
#     elif isinstance(element, dict):
#         return str(element)
#     else:
#         return str(element)

# def flatten_complex_cols(df):
#     for col in df.columns:
#         try:
#             if df[col].dtype == pl.List or df[col].dtype == pl.Struct:
#                 df = df.with_columns(
#                     pl.col(col).map_elements(flatten_complex_element).alias(col)
#                 )
#         except Exception as e:
#             print(f"Error processing column {col}: {str(e)}")
#             # If an error occurs, convert the column to string as a fallback
#             df = df.with_columns(pl.col(col).cast(pl.Utf8).alias(col))
#     return df

    
# def main():
#     base_dir=Path("/REDACTED_HPC_PATH/projects/myco/reads/spids/").absolute().resolve()
#     sacct_df = read_sacct_data("/REDACTED_HPC_PATH/projects/myco/slurmsout/sacct.out")
    
#     run_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

#     all_rp_times = pl.DataFrame()
#     all_command_tracks =  pl.DataFrame()
#     for run_id in run_dirs:
#         # print(run_id)
#         try:
#             bla = process_run_data(run_id)
#         except:
#             print(f"Failed to process {run_id}")
#             continue
#         if bla is None:
#             continue
#         new_time_data, new_merged_df = bla
#         all_rp_times = all_rp_times.vstack(new_time_data.with_columns(pl.lit(run_id).alias("run_id")))
#         all_command_tracks = all_command_tracks.vstack(new_merged_df.with_columns(pl.lit(run_id).alias("run_id")))
        
        
#     spid_data = pl.read_csv("/REDACTED_HPC_PATH/code/blits/mycovirome/metadata/merged_data.csv", separator=",",schema_overrides={"spid": pl.Utf8})   
#     spid_data = spid_data.drop("fastq").unique()
#     spid_data = spid_data.rename({"spid": "run_id"})
    
#     all_host_data = pl.read_csv("/REDACTED_HPC_PATH/code/blits/mycovirome/metadata/host_genome_stats.tsv", separator="\t")
#     all_host_data = all_host_data.select(pl.col("host_DNA_path"), pl.col("num_seqs"),pl.col("sum_len"), pl.col("GC(%)"),pl.col("max_len"),pl.col("avg_len"),pl.col("min_len"))
#     all_host_data = all_host_data.join(spid_data, on="host_DNA_path", how="left")
    
#     rp_times = all_rp_times.join(all_host_data, left_on="run_id", right_on="run_id", how="left")
#     rp_times = rp_times.join(sacct_df, on="run_id", how="left")
#     rp_times = rp_times.with_columns(pl.lit("isolate").alias("sample_type"))
#     flatten_complex_cols(rp_times).write_csv("rp_times.csv")
#     flatten_complex_cols(all_command_tracks).write_csv("all_command_tracks.csv")

#     # Convert total_runtime_from_timerr to seconds for easier plotting
#     rp_times = rp_times.with_columns(
#         pl.col("total_runtime_from_timerr").str.split(":").map_elements(
#             lambda x: int(x[0])*3600 + int(x[1])*60 + int(x[2]),
#             return_dtype=pl.Int64
#         ).alias("total_runtime_seconds")
#     )
    
#     rp_times = rp_times.with_columns(
#         pl.col("Percent of CPU this job got").str.replace("%", "").cast(pl.UInt16).alias("cpu_percent")
#     )
    
#     rp_times = rp_times.with_columns(
#         pl.col("Maximum resident set size (kbytes)").cast(pl.UInt64).alias("max_rss_mb") / 1e4
#     )

#     rp_times = rp_times.with_columns(
#         pl.col("Elapsed (wall clock) time (h:mm:ss or m:ss)").cast(pl.Utf8).str.replace(":", "").cast(pl.Float32()).alias("wall_time")
#     )

#     rp_times = rp_times.with_columns(
#         pl.col("System time (seconds)").cast(pl.Utf8).str.replace(":", "").cast(pl.Float32()).alias("system_time")
#     )

#     rp_times = rp_times.with_columns(   
#         pl.col("User time (seconds)").cast(pl.Utf8).str.replace(":", "").cast(pl.Float32()).alias("user_time")
#     )

#     rp_times = rp_times.with_columns(
#         pl.col("Percent of CPU this job got").str.replace("%", "").cast(pl.UInt16).alias("cpu_percent")
#     )
    
#     rp_times = rp_times.with_columns(
#         pl.col("Maximum resident set size (kbytes)").cast(pl.UInt64).alias("max_rss_mb") / 1e4
#     )
    
#     rp_times = rp_times.with_columns(
#         pl.col("Elapsed (wall clock) time (h:mm:ss or m:ss)").cast(pl.Utf8).str.replace(":", "").cast(pl.Float32()).alias("wall_time")
#     )
    
#     rp_times = rp_times.with_columns(
#         pl.col("System time (seconds)").cast(pl.Utf8).str.replace(":", "").cast(pl.Float32()).alias("system_time")
#     )
#     rp_times = rp_times.with_columns(
#         pl.col("User time (seconds)").cast(pl.Utf8).str.replace(":", "").cast(pl.Float32()).alias("user_time")
#     )
    


#     # comapre the sacct data with the timerr data
#     # first get the job ids in common
#     sacct_job_ids = sacct_df.select(pl.col("JobID")).to_series().to_list()
#     timerr_job_ids = rp_times.select(pl.col("JobID")).to_series().to_list()
#     job_ids_in_common = set(sacct_job_ids).intersection(set(timerr_job_ids))
#     # now get the data for the job ids in common
#     sacct_data = sacct_df.filter(pl.col("JobID").is_in(job_ids_in_common))
#     timerr_data = rp_times.filter(pl.col("JobID").is_in(job_ids_in_common))
    
#     # now let's plot the difference between the sacct time timerr time
#     # first convert the timerr time to seconds
#     timerr_data = timerr_data.with_columns(
#         pl.col("total_runtime_from_timerr").str.split(":").map_elements(
#             lambda x: int(x[0])*3600 + int(x[1])*60 + int(x[2]), return_dtype=pl.Int64
#         ).alias("total_runtime_seconds")
#     )
    
#     # now calculate the time difference
#     timerr_data = timerr_data.with_columns((
#         pl.col("total_runtime_seconds").cast(pl.Int64) - pl.col("elapsed_seconds").cast(pl.Int64)
#     ).alias("difference"))
    
#     # and the reported memory difference
#     timerr_data = timerr_data.with_columns((
#         pl.col("max_rss_mb").cast(pl.Int64) - (pl.col("MaxVMSize_B")*1e6).cast(pl.Int64)
#     ).alias("difference"))
    
#     # now plot the time difference
#     fig, ax = plt.subplots(figsize=(12, 8))
#     ax.hist(timerr_data["difference"].to_numpy(), bins=100)
#     ax.set_title("Difference between Timerr and Sacct")
#     ax.set_xlabel("Difference (seconds)")
#     ax.set_ylabel("Count")
#     plt.savefig("difference_between_timerr_and_sacct.png")
#     plt.close()
    
#     # now let's plot the difference between acct memory and timerr memory:
#     fig, ax = plt.subplots(figsize=(12, 8))
#     ax.hist(timerr_data["difference"].to_numpy(), bins=100)
#     ax.set_title("Difference between Timerr and Sacct")
#     ax.set_xlabel("Difference (seconds)")
#     ax.set_ylabel("Count")
#     plt.savefig("difference_between_timerr_and_sacct.png")
#     plt.close()
    
    
    
#     # now let's plot the total runtime as a function of the total number input reads and the host genome size:
#     import matplotlib.pyplot as plt
#     import seaborn as sns


#     # Plot 1: Total input reads and Host genome size, with facets for system time, user time, and wall time
#     fig, axes = plt.subplots(3, 1, figsize=(12, 24), sharex=True)
#     time_types = ["system_time", "user_time", "wall_time"]
#     titles = ["System Time", "User Time", "Wall Time"]

#     for i, (ax, time_type, title) in enumerate(zip(axes, time_types, titles)):


    
#     # now let's plot the total runtime as a function of the total number input reads and the host genome size:

#     # Import required libraries for plotting
#     import matplotlib.pyplot as plt
#     import seaborn as sns

#     # Plot 1: Total input reads and Host genome size, with facets for system time, user time, and wall time
#     fig, axes = plt.subplots(3, 1, figsize=(12, 24), sharex=True)
#     time_types = ["system_time", "user_time", "wall_time"]
#     titles = ["System Time", "User Time", "Wall Time"]

#     for i, (ax, time_type, title) in enumerate(zip(axes, time_types, titles)):
#         scatter = ax.scatter(
#             rp_times["total_reads"].to_numpy(),
#             rp_times[time_type].to_numpy(),
#             c=rp_times["sum_len"].to_numpy(),
#             s=rp_times["sum_len"].to_numpy() / 1e7,  # Adjust size scaling as needed
#             cmap="viridis",
#             alpha=0.7
#         )
#         ax.set_title(f"{title} vs. Input Reads and Host Genome Size")
#         ax.set_xlabel("Total Input Reads")
#         ax.set_ylabel(f"{title} (seconds)")
#         ax.set_xscale("log")
#         ax.set_yscale("log")
#         # Add node size legend
#         node_size_legend = ax.legend(*scatter.legend_elements("sizes", num=4), 
#                                      title="Host Genome Size", loc="upper left")
#         ax.add_artist(node_size_legend)
#     # plt.colorbar(scatter, ax=axes, label="Host Genome Size")
#     plt.tight_layout()
#     plt.savefig("runtime_vs_reads_and_genome_size.png")
#     plt.close()

#     # Plot xxx : Total runtime vs. Number of threads and CPU percent used
#     # rp_times.plot(x="threads_provided", y="total_runtime_seconds", c="Percent of CPU this job got", s="total_reads", cmap="coolwarm", alpha=0.7, xscale="log", yscale="log", title="Total Runtime vs. Threads and CPU Usage", xlabel="Number of Threads Provided", ylabel="Total Runtime (seconds)", colorbar_label="CPU Usage (%)")

#     # Plot 2: Total runtime vs.  CPU percent used and total reads
#     fig, ax = plt.subplots(figsize=(12, 8))
#     scatter = ax.scatter(
#         rp_times["cpu_percent"].to_numpy(),
#         rp_times["total_runtime_seconds"].to_numpy(),
#         c=rp_times["total_reads"].to_numpy(),
#         s=rp_times["total_reads"].to_numpy() / 1e7,  # Adjust size scaling as needed
#         cmap="coolwarm",
#         alpha=0.7
#     )
#     ax.set_title("Total Runtime vs. CPU Usage and Total Reads")
#     ax.set_xlabel("CPU Usage (%)")
#     ax.set_ylabel("Total Runtime (seconds)")
#     plt.colorbar(scatter, label="Total Reads ")
#     # add node size legend
#     node_size_legend = plt.legend(*scatter.legend_elements(), title="Total Reads ")
#     ax.add_artist(node_size_legend)
#     plt.savefig("runtime_vs_cpu_usage_and_reads.png")
#     plt.close()

#     # Plot 3: max_rss vs. total_reads amd host genome size
#     fig, ax = plt.subplots(figsize=(12, 8))
#     scatter = ax.scatter(
#         rp_times["total_reads"].to_numpy(),
#         rp_times["max_rss_mb"].to_numpy(), 
#         c=rp_times["sum_len"].to_numpy(), 
#         s=rp_times["sum_len"].to_numpy() / 1e7,  # Adjust size scaling as needed
#         cmap="viridis",
#         alpha=0.7
#     )   
#     ax.set_title("max_rss vs. Total Reads and Host Genome Size")
#     ax.set_xlabel("Total Reads")
#     ax.set_ylabel("max_rss (mb)")
#     plt.colorbar(scatter, label="Host Genome Size")
#     plt.savefig("max_rss_vs_reads_and_genome_size.png")
#     plt.close()