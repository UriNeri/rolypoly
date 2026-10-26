import polars as pl
import pytest

from rolypoly.utils.viz.genome_maps import (
    fasta_lengths,
    load_contig_lengths,
    load_run_stats,
    write_report_for_dir,
)


def test_report_uses_contig_id_map_length_for_renamed_contig(tmp_path):
    output_dir = tmp_path / "run"
    protein_dir = output_dir / "annotation_results" / "protein_annotation"
    assembly_dir = output_dir / "assembly"
    protein_dir.mkdir(parents=True)
    assembly_dir.mkdir()
    marker_table = protein_dir / "combined_annotations.tsv"
    mapping_path = assembly_dir / "contigs_id_map.tsv"
    report_path = output_dir / "genome_maps.html"

    pl.DataFrame(
        {
            "query_full_name": ["CID_01_1 # 5 # 80 # 1 # ID=CID_01_1"],
            "hmm_full_name": ["RdRp"],
            "source": ["rvmt"],
            "env_from": [1],
            "env_to": [20],
            "qlen": [25],
            "hmm_from": [1],
            "hmm_to": [20],
            "hmm_len": [20],
            "full_hmm_evalue": [1e-20],
            "full_hmm_score": [50.0],
            "hmm_cov": [1.0],
            "ali_len": [20],
            "q1": [1],
            "q2": [20],
        }
    ).write_csv(marker_table, separator="\t")
    pl.DataFrame(
        {
            "old_id": ["NODE_1_length_100_cov_5"],
            "new_id": ["CID_01"],
            "length": [100],
        }
    ).write_csv(mapping_path, separator="\t")

    assert load_contig_lengths(mapping_path) == {"CID_01": 100}
    write_report_for_dir(
        output_dir,
        report_path,
        marker_table=marker_table,
        with_stats=False,
        mark_best=False,
    )

    html = report_path.read_text()
    assert '"contig": "CID_01"' in html
    assert '"length": 100' in html
    assert '"raw_id": "NODE_1_length_100_cov_5"' in html


def test_fasta_lengths_streams_valid_records(tmp_path):
    fasta = tmp_path / "contigs.fa"
    fasta.write_text(">a\nACGT\n>b\nAA\n")

    assert fasta_lengths(fasta) == [4, 2]


def test_assembly_populations_use_each_source_and_sort_by_contig_count(tmp_path):
    assembly = tmp_path / "assembly"
    assembly.mkdir()
    populations = {
        "all_contigs.fasta": [10, 6, 4, 2],
        "dereplicated_contigs.fasta": [10, 6, 4],
        "filtered_assembly.fasta": [10, 6],
        "clustered_assembly.fasta": [10],
        "length_filtered.fasta": [6, 4],
    }
    for name, lengths in populations.items():
        (assembly / name).write_text(
            "".join(f">c{i}\n{'A' * length}\n" for i, length in enumerate(lengths))
        )
    (tmp_path / "all_matched_contigs.fasta").write_text(">viral_candidate\nAAAA\n")
    (assembly / "contigs_id_map.tsv").write_text(
        "new_id\tassembler\tlength\n"
        "c1\tmegahit\t10\nc2\tmegahit\t6\nc3\tspades\t4\nc4\tspades\t2\n"
    )
    # Protein FASTAs and arbitrary input files must not masquerade as assemblies.
    (tmp_path / "predicted_orfs.faa").write_text(">orf\nMPEPTIDE\n")
    (tmp_path / "unrelated.fasta").write_text(">other\nACGT\n")
    rows = load_run_stats(tmp_path)["assembly_sources"]
    assert [row["n_contigs"] for row in rows] == [4, 3, 2, 2, 2, 2, 1, 1]
    by_source = {row["source"]: row for row in rows if "ID map" not in row["name"]}
    raw = by_source["assembly/all_contigs.fasta"]
    assert {key: raw[key] for key in ("n_contigs", "total_bp", "n50", "min", "max", "mean")} == {
        "n_contigs": 4, "total_bp": 22, "n50": 6, "min": 2, "max": 10, "mean": 5.5,
    }
    selected = by_source["all_matched_contigs.fasta"]
    assert selected["n_contigs"] == 1
    assert selected["total_bp"] == selected["n50"] == 4
    assert "candidates (not confirmed)" in selected["name"]
    assert "predicted_orfs.faa" not in by_source
    assert "unrelated.fasta" not in by_source


def test_assembly_sources_work_without_an_id_map_and_include_empty_candidates(tmp_path):
    (tmp_path / "length_filtered.fasta").write_text(">c1\nAAAAAA\n")
    (tmp_path / "all_matched_contigs.fasta").write_text("")
    stats = load_run_stats(tmp_path)
    rows = stats["assembly_sources"]
    assert [row["n_contigs"] for row in rows] == [1, 0]
    empty = rows[1]
    assert empty["state"] == "available"
    assert empty["total_bp"] == 0
    assert all(empty[key] is None for key in ("n50", "min", "max", "mean"))
    assert not any("raw assembly" in row["name"] for row in rows)
    report = write_report_for_dir(tmp_path, initial_tab="stats")
    assert report is not None
    html = report.read_text()
    assert 'id="assemblyStatsTable"' in html
    assert "N/A - empty population" in html
    assert "<th>Minimum (bp)</th>" in html
    assert '<div class="l">N50</div>' not in html


def test_assembly_sources_show_unreadable_artifacts_after_available_rows(tmp_path, caplog):
    (tmp_path / "length_filtered.fasta").write_text(">c1\nAAAA\n")
    (tmp_path / "clustered_assembly.fasta").symlink_to(tmp_path / "missing.fasta")
    rows = load_run_stats(tmp_path)["assembly_sources"]
    assert rows[0]["source"] == "length_filtered.fasta"
    assert rows[1]["source"] == "clustered_assembly.fasta"
    assert rows[1]["state"] == "unavailable"
    assert "Unavailable" in rows[1]["note"]
    assert "could not read FASTA" in caplog.text


def test_assembly_sources_deduplicate_supplied_input_map_aliases(tmp_path):
    (tmp_path / "contigs_id_map.tsv").write_text(
        "old_id\tnew_id\tassembler\tlength\n"
        "original1\tc1\tinput\t10\noriginal2\tc1\tinput\t10\n"
        "original3\tc2\tinput\t6\n"
    )
    stats = load_run_stats(tmp_path)
    assert stats["assembly_sources"] == [{
        "name": "User-supplied assembly (ID map)", "source": "contigs_id_map.tsv",
        "state": "available", "n_contigs": 2, "total_bp": 16,
        "n50": 10, "max": 10, "min": 6, "mean": 8.0,
    }]
    assert stats["assembly"]["n_contigs"] == 2


def test_assembly_sources_recover_pruned_assemblers_and_available_raw_fastas(tmp_path):
    megahit = tmp_path / "megahit_custom_out"
    megahit.mkdir()
    (megahit / "final.contigs.fa").write_text(">m1\nAAAA\n")
    (tmp_path / "contigs_id_map.tsv").write_text(
        "new_id\tassembler\tlength\nm1\tmegahit\t4\ns1\tspades\t10\n"
    )
    (tmp_path / "dereplicated_contigs.fasta").write_text(">s1\nAAAAAAAAAA\n")
    (tmp_path / "final_assembly.fasta").symlink_to(tmp_path / "dereplicated_contigs.fasta")
    rows = load_run_stats(tmp_path)["assembly_sources"]
    assert len(rows) == 5
    assert sum("megahit" in row["name"] for row in rows) == 1
    assert any(row["name"] == "Assembler: spades (raw, ID map)" for row in rows)
    alias = next(row for row in rows if row["source"] == "final_assembly.fasta")
    assert "Alias of dereplicated_contigs.fasta" in alias["note"]


def test_assembly_sources_include_distinct_spades_outputs(tmp_path):
    for directory, filename, sequence in (
        ("spades_rna_output", "transcripts.fasta", "AAAA"),
        ("spades_rnaviral_output", "scaffolds.fasta", "AAAAAA"),
    ):
        source = tmp_path / directory
        source.mkdir()
        (source / filename).write_text(f">c1\n{sequence}\n")
    (tmp_path / "contigs_id_map.tsv").write_text(
        "new_id\tassembler\tlength\nc1\tspades\t4\nc2\tspades_rnaviral\t6\n"
    )
    rows = load_run_stats(tmp_path)["assembly_sources"]
    assert len(rows) == 3
    assert len([row for row in rows if row["name"].startswith("Assembler:")]) == 2
    assert {row["total_bp"] for row in rows} == {4, 6, 10}


def test_load_run_stats_falls_back_to_id_map_when_endpoint_fasta_unreadable(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "run"
    assembly_dir = output_dir / "assembly"
    assembly_dir.mkdir(parents=True)
    pl.DataFrame(
        {
            "old_id": ["raw1", "raw2"],
            "new_id": ["CID_1", "CID_2"],
            "assembler": ["input", "input"],
            "length": [100, 50],
        }
    ).write_csv(assembly_dir / "contigs_id_map.tsv", separator="\t")
    (assembly_dir / "length_filtered.fasta").write_text(">CID_1\nACGT\n")

    def fail_parse(_path):
        raise ValueError("bad fasta")

    monkeypatch.setattr(
        "rolypoly.utils.viz.genome_maps.parse_fastx_file", fail_parse
    )

    stats = load_run_stats(output_dir)
    assert stats["assembly"]["source"] == "contigs_id_map.tsv"
    assert stats["assembly"]["n_contigs"] == 2
    assert stats["assembly"]["total_bp"] == 150


@pytest.mark.parametrize("invalid_length", ["broken", "", "12.5", "-2", "0"])
def test_run_stats_rejects_incomplete_assembly_totals(
    tmp_path, caplog, invalid_length
):
    mapping = tmp_path / "contigs_id_map.tsv"
    mapping.write_text(
        "new_id\tassembler\tlength\n"
        f"CID_1\tmegahit\t100\nCID_2\tspades\t{invalid_length}\n"
    )
    # Other valid statistics must survive a malformed optional assembly map.
    (tmp_path / "stats_host.txt").write_text("#Total\t10\n#Matched\t2\t20%\n")

    stats = load_run_stats(tmp_path)

    assert stats["assembly"] == {}
    assert stats["reads"][0]["kept"] == 8
    warnings = [record for record in caplog.records if record.levelname == "WARNING"]
    assert len(warnings) == 1
    assert str(mapping) in warnings[0].getMessage()

    report = write_report_for_dir(
        tmp_path,
        extra_tabs=[
            {"id": "sample", "label": "Sample", "columns": ["sample"], "rows": [["test"]]}
        ],
    )
    assert report is not None
    assert '"assembly": {}' in report.read_text()
    assert '"kept": 8' in report.read_text()


def test_run_stats_keeps_final_assembly_when_mapping_lengths_are_invalid(
    tmp_path, caplog
):
    mapping = tmp_path / "contigs_id_map.tsv"
    mapping.write_text("new_id\tassembler\tlength\nCID_1\tmegahit\tbroken\n")
    (tmp_path / "length_filtered.fasta").write_text(">CID_1\nACGT\n")

    stats = load_run_stats(tmp_path)

    assert stats["assembly"]["source"] == "length_filtered.fasta"
    assert stats["assembly"]["total_bp"] == 4
    assert stats["assembly"]["n_contigs"] == 1
    assert "assemblers" not in stats["assembly"]
    assert str(mapping) in caplog.text


def test_run_stats_final_assembly_keeps_raw_assembler_breakdown(tmp_path):
    (tmp_path / "contigs_id_map.tsv").write_text(
        "new_id\tassembler\tlength\n"
        "CID_1\tmegahit\t100\nCID_2\tspades\t50\nCID_3\tspades\t200\n"
        "CID_4\t\t25\n"
    )
    (tmp_path / "length_filtered.fasta").write_text(">CID_1\nACGT\n")

    stats = load_run_stats(tmp_path)

    assert stats["assembly"]["source"] == "length_filtered.fasta"
    assert stats["assembly"]["total_bp"] == 4
    assert [
        (item["name"], item["n_contigs"], item["total_bp"])
        for item in stats["assembly"]["assemblers"]
    ] == [("megahit", 1, 100), ("spades", 2, 250), ("unknown", 1, 25)]


def test_run_stats_mapping_without_assembler_keeps_raw_totals(tmp_path):
    (tmp_path / "contigs_id_map.tsv").write_text(
        "new_id\tlength\nCID_1\t100\nCID_2\t50\n"
    )

    stats = load_run_stats(tmp_path)

    assert stats["assembly"]["source"] == "contigs_id_map.tsv"
    assert stats["assembly"]["n_contigs"] == 2
    assert stats["assembly"]["total_bp"] == 150


def test_run_stats_missing_optional_inputs_are_quiet(tmp_path, caplog):
    assert load_run_stats(tmp_path) is None
    assert not caplog.records


@pytest.mark.parametrize("prefix", ["", "sample.fq_"])
def test_report_reads_falco_single_and_multiple_input_layouts(tmp_path, prefix):
    qc = tmp_path / "run_info" / "falco_post_trim_reads"
    qc.mkdir(parents=True)
    (qc / f"{prefix}fastqc_data.txt").write_text(
        "##Falco\t1.3.1\n>>Basic Statistics\tpass\n#Measure\tValue\n"
        "Filename\tsample.fq\nTotal Sequences\t20\nTotal Bases\t1600 bp\n"
        "Sequence length\t80\n%GC\t50.0\n>>END_MODULE\n"
    )
    (qc / f"{prefix}summary.txt").write_text("PASS\tBasic Statistics\tsample.fq\n")
    (qc / f"{prefix}fastqc_report.html").write_text("<html>Falco plots</html>")

    stats = load_run_stats(tmp_path)

    assert stats["falco"] == [{
        "file": "sample.fq", "total_sequences": 20, "total_bases": "1600 bp",
        "gc": 50.0, "length": "80", "flags": {"Basic Statistics": "PASS"},
        "report_html": "<html>Falco plots</html>",
    }]
    report = write_report_for_dir(tmp_path, extra_tabs=[{
        "id": "sample", "label": "Sample", "columns": ["sample"], "rows": [["test"]]
    }])
    assert '"total_sequences": 20' in report.read_text()
    assert "Falco plots" in report.read_text()


@pytest.mark.parametrize("contents", [
    "", "#Total\t10\n", "#Total\tbad\n#Matched\t2\n",
    "#Total\t10\n#Matched\t11\n", "#Total\t10\n#Matched\t-1\n",
])
def test_run_stats_warns_and_omits_invalid_bbduk_file(tmp_path, caplog, contents):
    invalid = tmp_path / "stats_bad.txt"
    invalid.write_text(contents)
    (tmp_path / "stats_good.txt").write_text("#Total\t10\n#Matched\t2\t20%\n")

    stats = load_run_stats(tmp_path)

    assert [(row["step"], row["kept"]) for row in stats["reads"]] == [("good", 8)]
    assert str(invalid) in caplog.text


@pytest.mark.parametrize("contents", [
    "", ">>Basic Statistics\tpass\nFilename\tsample.fq\nTotal Sequences\t20\n",
    ">>Basic Statistics\tpass\nTotal Sequences\tbad\n>>END_MODULE\n",
    ">>Basic Statistics\tpass\nTotal Sequences\t-1\n>>END_MODULE\n",
    ">>Basic Statistics\tpass\nTotal Sequences\t20\n%GC\tNaN\n>>END_MODULE\n",
])
def test_run_stats_warns_and_omits_invalid_falco_file(tmp_path, caplog, contents):
    invalid = tmp_path / "bad_fastqc_data.txt"
    invalid.write_text(contents)
    (tmp_path / "good_fastqc_data.txt").write_text(
        ">>Basic Statistics\tpass\nFilename\tempty.fq\n"
        "Total Sequences\t0\n%GC\t0\n>>END_MODULE\n"
    )

    stats = load_run_stats(tmp_path)

    assert len(stats["falco"]) == 1
    assert stats["falco"][0]["total_sequences"] == 0
    assert str(invalid) in caplog.text


def test_run_stats_bad_falco_siblings_preserve_basic_statistics(tmp_path, caplog):
    (tmp_path / "sample_fastqc_data.txt").write_text(
        ">>Basic Statistics\tpass\nTotal Sequences\t20\n>>END_MODULE\n"
    )
    summary = tmp_path / "sample_summary.txt"
    summary.write_text("PASS\tBasic Statistics\tsample.fq\ntruncated\n")
    report = tmp_path / "sample_fastqc_report.html"
    report.write_bytes(b"\xff")

    stats = load_run_stats(tmp_path)

    assert stats["falco"][0]["total_sequences"] == 20
    assert stats["falco"][0]["flags"] == {}
    assert stats["falco"][0]["report_html"] == ""
    assert str(summary) in caplog.text
    assert str(report) in caplog.text


def test_run_stats_reads_quoted_output_tracker_csv(tmp_path):
    from rolypoly.utils.logging.output_tracker import OutputTracker

    tracker = OutputTracker()
    tracker.df = pl.DataFrame({
        "filename": ['sample, "one".fq'],
        "command": ['bbduk ref="one,two"\ncontinued'],
        "command_name": ["filter,reads"], "file_type": ["fastq"],
        "file_size": [0], "is_merged": [True],
    })
    tracker.to_csv(tmp_path / "output_tracker.csv")

    stats = load_run_stats(tmp_path)

    assert stats["files"] == [{"step": "filter,reads", "type": "fastq", "size": 0, "merged": True}]


@pytest.mark.parametrize("bad_size", ["broken", "", "-1", "1.5"])
def test_run_stats_invalid_tracker_does_not_publish_partial_rows(tmp_path, caplog, bad_size):
    first = tmp_path / "a"
    second = tmp_path / "b"
    first.mkdir()
    second.mkdir()
    invalid = first / "output_tracker.csv"
    invalid.write_text(
        "command_name,file_type,file_size,is_merged\n"
        f"first,fastq,10,true\nsecond,fastq,{bad_size},false\n"
    )
    (second / "output_tracker.csv").write_text(
        "command_name,file_type,file_size,is_merged\nvalid,fastq,20,false\n"
    )

    stats = load_run_stats(tmp_path)

    assert stats["files"] == [{"step": "valid", "type": "fastq", "size": 20, "merged": False}]
    assert str(invalid) in caplog.text


@pytest.mark.parametrize("bad_pct", ["broken%", "", "NaN", "inf", "-1%", "101%"])
def test_run_stats_bad_rrna_percentage_omits_domain_totals(tmp_path, caplog, bad_pct):
    path = tmp_path / "stats_rrna.txt"
    path.write_text(
        "#Total\t100\n#Matched\t20\t20%\n#Name\tReads\tReadsPct\n"
        f"reference1\t10\t10%\nreference2\t10\t{bad_pct}\n"
    )

    stats = load_run_stats(tmp_path)

    assert stats["reads"][0]["kept"] == 80
    assert stats["rrna_domain"] == {}
    assert stats["rrna_top"] == []
    assert str(path) in caplog.text


def test_run_stats_rrna_domain_percentages_include_zero(tmp_path):
    (tmp_path / "stats_rrna.txt").write_text(
        "#Total\t100\n#Matched\t15\t15%\n#Name\tReads\tReadsPct\n"
        "1@SILVA@SSU_eukaryote_rRNA@a\t10\t10%\n"
        "2@NCBI@LSU_prokaryote_rRNA@b\t5\t5%\nreference\t0\t0%\n"
    )

    stats = load_run_stats(tmp_path)

    assert stats["rrna_domain"] == {"eukaryotic": 10.0, "prokaryotic": 5.0, "unknown": 0.0}
    assert [row["pct"] for row in stats["rrna_top"]] == ["10%", "5%", "0%"]
