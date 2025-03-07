import os
import shutil
import re
import polars as pl

# Define the source directory containing the output files
source_dir = '/REDACTED_HPC_PATH/projects/myco/slurmsout/filter_reads/'

# Define the base directory containing the run folders
base_run_dir = '/REDACTED_HPC_PATH/projects/myco/reads/spids/'

# Regular expression to extract the run ID from the file content
run_id_pattern = r'/spids/(\d+)/\1_filtered_reads'

def get_run_id(file_path):
    with open(file_path, 'r') as f:
        for _ in range(5):  # Read first 5 lines
            line = f.readline()
            match = re.search(run_id_pattern, line)
            if match:
                return match.group(1)
    return None

def main():
    slurm_out_df = pl.read_csv("/REDACTED_HPC_PATH/projects/myco/slurmsout/sacct.out",separator="|")
    slurm_out_df = slurm_out_df.filter(pl.col("JobID").str.contains("_"))
    # filter all jobs that failed (we want to remove them completely, so we need to first get all jobids that failed, then remove them from the slurm_out_df):
    failed_jobs = slurm_out_df.filter(pl.col("ExitCode")!="0:0").select(pl.col("JobID")).to_series().to_list()
    slurm_out_df = slurm_out_df.filter(~pl.col("JobID").is_in(failed_jobs))
    
    # slurm_out_df.write_csv("/REDACTED_HPC_PATH/projects/myco/slurmsout/sacct_filtered.out",separator="|")
    tmp_dict = {}
    for filename in os.listdir(source_dir):
        file_path = os.path.join(source_dir, filename)
        
        if os.path.isfile(file_path):
            run_id = get_run_id(file_path)
            
            if run_id:
                # dest_dir = os.path.join(base_run_dir, f'{run_id}_filtered_reads', 'run_info')
                
                # if os.path.exists(dest_dir):
                    # dest_path = os.path.join(dest_dir, filename)
                    # shutil.copy2(file_path, dest_path)
                tmp_dict[run_id] = filename.replace("filter_reads-","").replace(".out","")
                    # print(f"Copied {filename} to {dest_path}")
                # else:
                    # print(f"Destination directory not found for run ID {run_id}")
            else:
                print(f"Could not find run ID in {filename}")
                
    tmp_df = pl.DataFrame({"run_id":tmp_dict.keys(),"JobID":tmp_dict.values()})
    tmp_df.write_csv("/REDACTED_HPC_PATH/projects/myco/slurmsout/filter_reads/run_id_sample_id.csv")
    slurm_out_df = slurm_out_df.join(tmp_df,on="JobID",how="left")
    slurm_out_df.write_csv("/REDACTED_HPC_PATH/projects/myco/slurmsout/sacct_filtered_with_sample_id.csv",separator=",")
    
    
    # Plot: SLURM Elapsed Time vs Total Reads and Allocated CPUs
    import matplotlib.pyplot as plt
    rp_times = slurm_out_df.select(pl.col("total_reads"),pl.col("elapsed_seconds"),pl.col("AllocCPUS"),pl.col("MaxRSS")).filter(pl.col("total_reads").is_not_null())

    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(
        rp_times["total_reads"].to_numpy(),
        rp_times["elapsed_seconds"].to_numpy(),
        c=rp_times["AllocCPUS"].to_numpy(),
        s=rp_times["MaxRSS"].to_numpy() / 1e5,  # Adjust size scaling as needed
        cmap="viridis",
        alpha=0.7
    )
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title("SLURM Elapsed Time vs Total Reads and Allocated CPUs")
    ax.set_xlabel("Total Reads")
    ax.set_ylabel("Elapsed Time (seconds)")
    plt.colorbar(scatter, label="Allocated CPUs")
    plt.savefig("slurm_elapsed_vs_reads_and_cpus.png")
    plt.close()
        
    
if __name__ == "__main__":
    main()