# # setup.py
# from setuptools import setup, find_packages, Extension
# from setuptools.command.develop import develop
# from setuptools.command.install import install
# from setuptools.command.build_ext import build_ext
# import os
# import subprocess
# from pathlib import Path
# import shutil
# import json
# import requests
# import tarfile
# import concurrent.futures
# import pgzip

# only_missing = os.getenv("rp_only_missing", "False").lower() == "true"
# skip_external_dependencies = os.getenv("rp_skip_external_dependencies", "False").lower() == "true"
# force_clean = os.getenv("rp_force_remove_ext", "True").lower() == "true"

# # Classes 
# class ExternalSoftware:
#     def __init__(self, name, url, build_cmd, executable):
#         self.name = name
#         self.url = url
#         self.build_cmd = build_cmd
#         self.executable = executable

# class CustomBuild(build_ext):
#     def run(self):
#         external_deps_dir = Path(__file__).parent / 'external_dependencies'
#         if force_clean:
#             shutil.rmtree(external_deps_dir, ignore_errors=True)
#         external_deps_dir.mkdir(exist_ok=True)
        
#         software_list = [
#             ExternalSoftware(
#                 name="pigz",
#                 url="https://github.com/madler/pigz/archive/refs/tags/v2.8.tar.gz",
#                 build_cmd="make",
#                 executable="pigz"
#             )
#         ]
        
#         if only_missing:
#             software_list = [software for software in software_list if not shutil.which(software.name)]
        
#         if not skip_external_dependencies and software_list:
#             for software in software_list:
#                 print(f"Building {software.name}    ")
#                 extract_dir = external_deps_dir / f'{software.name}_source_code'
#                 extract_dir.mkdir(exist_ok=True)
#                 build_dir = download_and_extract(software.url, extract_dir)
#                 build_software(software=software, build_dir=build_dir[0], install_dir=external_deps_dir)

#         # Run the standard build_ext
#         build_ext.run(self)
        
#         # Call download_external_dependencies here
#         if not skip_external_dependencies:
#             download_external_dependencies()

# class PostDevelopCommand(develop):
#     def run(self):
#         develop.run(self)

# class PostInstallCommand(install):
#     def run(self):
#         install.run(self)

# # Functions
# def is_linux_binary(file_path):
#     try:
#         with open(file_path, 'rb') as file:
#             return file.read(4) == b'\x7fELF'
#     except IOError:
#         return False

# def download_and_extract(url, extract_dir):
#     filename = Path(url).name
#     subprocess.check_call(["wget", url])
#     if filename.endswith(('.tar.gz', '.tgz')):
#         subprocess.check_call(["tar", "xzvf", filename, "-C", str(extract_dir)])
#     elif filename.endswith('.zip'):
#         subprocess.check_call(["unzip", filename, "-d", str(extract_dir)])
#     Path(filename).unlink()
#     return [d for d in extract_dir.iterdir() if d.is_dir()]

# # def update_config_file(external_deps_dir):
# #     config_file = Path(__file__).parent.parent / 'rp_config.json'
# #     if config_file.exists():
# #         with config_file.open('r') as f:
# #             config = json.load(f)
# #         config['external_deps_dir'] = str(external_deps_dir)
# #         with config_file.open('w') as f:
# #             json.dump(config, f, indent=4)
# #     else:
# #         print("rp_config.json not found. Skipping config update.")

# def download_file(url, filename):
#     with requests.get(url, stream=True) as r:
#         r.raise_for_status()
#         with open(filename, 'wb') as f:
#             for chunk in r.iter_content(chunk_size=8192):
#                 f.write(chunk)

# def extract_tar_gz(filename, extract_dir):
#     import pgzip
#     with pgzip.open(filename, 'rb', thread=max(1,os.cpu_count()//2)) as f_in:
#         with tarfile.open(fileobj=f_in, mode='r|') as tar:
#             tar.extractall(path=extract_dir)

# def process_dependency(dep, depen_dir):
#     name, url = dep
#     print(f"Installing {name}    ")
#     filename = Path(url).name
#     file_path = depen_dir / filename
    
#     # Download
#     download_file(url, file_path)
    
#     # Extract,,, if needed
#     if filename.endswith(('.tar.gz', '.tgz')):
#         extract_tar_gz(file_path, depen_dir)
#         file_path.unlink()  # Remove the archive after extraction
#     elif filename.endswith('.zip'):
#         shutil.unpack_archive(file_path, depen_dir)
#         file_path.unlink()  # Remove the archive after extraction
    
#     return name


# def download_external_dependencies():
#     depen_dir = Path(__file__).parent / 'external_dependencies'
#     depen_dir.mkdir(exist_ok=True)

#     dependencies = [
#         ("spades", "https://github.com/ablab/spades/releases/download/v4.0.0/SPAdes-4.0.0-Linux.tar.gz"),
#         ("seqkit", "https://github.com/shenwei356/seqkit/releases/download/v2.8.2/seqkit_linux_amd64.tar.gz"),
#         ("datasets", "https://ftp.ncbi.nlm.nih.gov/pub/datasets/command-line/v2/linux-amd64/datasets"),
#         ("dataformat", "https://ftp.ncbi.nlm.nih.gov/pub/datasets/command-line/v2/linux-amd64/dataformat"),
#         ("bbmap", "  https://sourceforge.net/projects/bbmap/files/BBMap_39.06.tar.gz"),
#         ("megahit", "https://github.com/voutcn/megahit/releases/download/v1.2.9/MEGAHIT-1.2.9-Linux-x86_64-static.tar.gz"),
#         ("fastqc", " https://www.bioinformatics.babraham.ac.uk/projects/fastqc/fastqc_v0.12.1.zip"),
#         ("mmseqs"   , "https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz"),
#         ("plass", "https://mmseqs.com/plass/plass-linux-avx2.tar.gz"),
#         ("blast", "https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/2.15.0/ncbi-blast-2.15.0+-x64-linux.tar.gz"),
#         ("rush", "https://github.com/shenwei356/rush/releases/download/v0.5.4/rush_linux_amd64.tar.gz"),
#         ("diamond", "https://github.com/bbuchfink/diamond/releases/download/v2.1.9/diamond-linux64.tar.gz") 
#     ]

#     # Use ThreadPoolExecutor to download and extract concurrently
#     with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
#         futures = [executor.submit(process_dependency, dep, depen_dir) for dep in dependencies]
#         for future in concurrent.futures.as_completed(futures):
#             name = future.result()
#             print(f"Finished installing {name}")

#     print("Collecting bin paths    ")
#     bin_paths = [str(depen_dir)]  # Add the main directory
#     for item in depen_dir.iterdir():
#         if item.is_dir():
#             bin_paths.append(str(item))  # Add first-degree subdirectories
#             # for subitem in item.iterdir():
#             #     if subitem.is_dir():
#             #         bin_paths.append(str(subitem))  # Add second-degree subdirectories
#         else:
#             print(f"Making {item.name} executable    ")
#             item.chmod(item.stat().st_mode | 0o111)  # Add execute permission

#     # print("Cleaning up    ")
#     # for file in depen_dir.glob('*.gz'):
#     #     file.unlink()
#     # for file in depen_dir.glob('*.bz2'):
#     #     file.unlink()
#     # for file in depen_dir.glob('*.zip'):
#     #     file.unlink()

#     # Update the JSON configuration file
#     config_path = Path(__file__).parent / 'rpconfig.json'
#     if config_path.exists():
#         with config_path.open('r') as f:
#             config = json.load(f)
#     else:
#         config = {'external_deps_dir': [""]}

#     config['external_deps_dir'] = ':'.join(bin_paths)

#     with config_path.open('w') as f:
#         json.dump(config, f, indent=4)

#     # os.chdir(cwd)

# def build_software(software, build_dir, install_dir):
#     cwd = Path.cwd()
#     os.chdir(build_dir)
#     subprocess.check_call(software.build_cmd, shell=True)
#     os.chdir(cwd)
#     src = build_dir / software.executable
#     dst = install_dir / software.executable
#     if src.exists():
#         shutil.copy2(src=src, dst=dst)
#         print(f"Making {software.name} executable    ")
#         dst.chmod(dst.stat().st_mode | 0o111)  # Add execute permission
#     else:
#         print(f"Warning: {software.executable} not found in {build_dir}")

# # def download_external_dependencies():
# #     depen_dir = Path(__file__).parent / 'external_dependencies'
# #     depen_dir.mkdir(exist_ok=True)
# #     cwd = Path.cwd()
# #     os.chdir(depen_dir)

# #     dependencies = [
# #         ("spades", "wget --quiet https://github.com/ablab/spades/releases/download/v4.0.0/SPAdes-4.0.0-Linux.tar.gz && tar xvfz SPAdes-4.0.0-Linux.tar.gz"),
# #         ("seqkit", "wget --quiet https://github.com/shenwei356/seqkit/releases/download/v2.8.2/seqkit_linux_amd64.tar.gz  && tar xvfz seqkit_linux_amd64.tar.gz"),
# #         ("datasets", "wget --quiet https://ftp.ncbi.nlm.nih.gov/pub/datasets/command-line/v2/linux-amd64/datasets"),
#         # ("dataformat", "wget --quiet https://ftp.ncbi.nlm.nih.gov/pub/datasets/command-line/v2/linux-amd64/dataformat"),
#         # ("bbmap", "wget --quiet  https://sourceforge.net/projects/bbmap/files/BBMap_39.06.tar.gz && tar xvfz BBMap_39.06.tar.gz"),
#         # ("megahit", "wget --quiet https://github.com/voutcn/megahit/releases/download/v1.2.9/MEGAHIT-1.2.9-Linux-x86_64-static.tar.gz && tar zvxf MEGAHIT-1.2.9-Linux-x86_64-static.tar.gz"),
#         # ("fastqc", "wget --quiet https://www.bioinformatics.babraham.ac.uk/projects/fastqc/fastqc_v0.12.1.zip && unzip fastqc_v0.12.1.zip"),
#         # ("mmseqs"   , "wget --quiet https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz && tar xvfz mmseqs-linux-avx2.tar.gz"),
#         # ("plass", "wget --quiet https://mmseqs.com/plass/plass-linux-avx2.tar.gz && tar xvfz plass-linux-avx2.tar.gz"),
#         # ("blast", "wget --quiet https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/2.15.0/ncbi-blast-2.15.0+-x64-linux.tar.gz && tar xvfz ncbi-blast-2.15.0+-x64-linux.tar.gz"),
#         # ("rush", "wget --quiet https://github.com/shenwei356/rush/releases/download/v0.5.4/rush_linux_amd64.tar.gz && tar xvfz rush_linux_amd64.tar.gz"),
#         # ("diamond", "wget --quiet https://github.com/bbuchfink/diamond/releases/download/v2.1.9/diamond-linux64.tar.gz && tar xvfz diamond-linux64.tar.gz") 
# #     ]   
# #     bin_paths = []

# #     for name, command in dependencies:
# #         print(f"Installing {name}    ")
# #         subprocess.run(command, shell=True, check=True)

# #     print("Cleaning up    ")
# #     for file in depen_dir.glob('*.gz'):
# #         file.unlink()
# #     for file in depen_dir.glob('*.bz2'):
# #         file.unlink()
# #     for file in depen_dir.glob('*.zip'):
# #         file.unlink()

# #     print("Collecting Linux binaries    ")
# #     binary_paths = []
    
# #     # for root, dirs, files in os.walk(depen_dir):
# #     #     for dir_name in dirs:
# #     #         if dir_name == "bin":
# #     #             bin_dir = Path(root) / dir_name
# #     #             bin_paths.append(str(bin_dir))
# #     #             for bin_file in bin_dir.iterdir():
# #     #                 if bin_file.is_file() and is_linux_binary(bin_file):
# #     #                     shutil.copy(bin_file, depen_dir)
# #     #                     binary_paths.append(str(depen_dir / bin_file.name))
        
# #     #     for file_name in files:
# #     #         full_path = Path(root) / file_name
# #     #         if full_path.is_file() and os.access(full_path, os.X_OK) and is_linux_binary(full_path):
# #     #             binary_paths.append(str(full_path))

# #     bin_paths = [str(depen_dir)]  # Add the main directory
# #     for item in depen_dir.iterdir():
# #         if item.is_dir():
# #             bin_paths.append(str(item))  # Add first-degree subdirectories
# #             for subitem in item.iterdir():
# #                 if subitem.is_dir():
# #                     bin_paths.append(str(subitem))  # Add second-degree subdirectories

    
# #     print("Making Linux binaries executable    ")
# #     for binary in binary_paths:
# #         Path(binary).chmod(0o755)

# #     # Update the JSON configuration file
# #     config_path = Path(__file__).parent / 'rpconfig.json'
# #     if config_path.exists():
# #         with config_path.open('r') as f:
# #             config = json.load(f)
# #     else:
# #         config = {'external_deps_dir': [""]}

# #     config['external_deps_dir'] = ':'.join(bin_paths)

# #     with config_path.open('w') as f:
# #         json.dump(config, f, indent=4)

# #     os.chdir(cwd)
    
#     # Actual execution
# setup(
#     name="rolypoly",
#     version="0.1",
#     packages=find_packages(),
#     include_package_data=True,
#     install_requires=[
#         "rich_click",
#         "pyhmmer",
#         "pyrodigal-gv",
#         "pgzip",
#         "polars",
#         "Bio"
#     ],
#     cmdclass={
#         'build_ext': CustomBuild,
#         'develop': PostDevelopCommand,
#         'install': PostInstallCommand,
#     },
#     ext_modules=[],
#     entry_points={
#         "console_scripts": [
#             "rolypoly=rolypoly:rolypoly",
#         ],
#     },
# )


