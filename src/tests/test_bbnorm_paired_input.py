"""Keep BBNorm's repeated passes away from the separate-mate reader."""

from pathlib import Path
from unittest.mock import patch
import gzip
import random
import re
import shutil
import subprocess

import pytest

from rolypoly.utils.bio.library_detection import create_sample_file


@pytest.mark.parametrize("fail", [False, True])
def test_paired_normalization_uses_interleaved_passes_and_cleans_up(tmp_path, fail):
    calls = []

    def reformat(**kwargs):
        calls.append(kwargs)
        if "out" in kwargs:
            Path(kwargs["out"]).write_text("fixture")
        else:
            assert Path(kwargs["in1"]).read_text() == "normalized"

    def normalize(**kwargs):
        assert Path(kwargs["in1"]).exists()
        assert kwargs["interleaved"] == "t"
        assert "in2" not in kwargs and "out2" not in kwargs
        assert kwargs["target"] == 17 and kwargs["min"] == 3
        if fail:
            raise RuntimeError("normalization failed")
        Path(kwargs["out"]).write_text("normalized")
        return "", ""

    with patch("bbmapy.reformat", side_effect=reformat), patch(
        "bbmapy.bbnorm", side_effect=normalize
    ):
        kwargs = dict(
            file_path="reads_R1.fq.gz,reads_R2.fq.gz",
            output_file=f"{tmp_path}/out_R1.fq,{tmp_path}/out_R2.fq",
            subset_type="bbnorm", sample_size=17, bbnorm_min_depth=3,
        )
        if fail:
            with pytest.raises(RuntimeError, match="normalization failed"):
                create_sample_file(**kwargs)
        else:
            create_sample_file(**kwargs)
            assert calls[1]["out1"] == str(tmp_path / "out_R1.fq")
            assert calls[1]["out2"] == str(tmp_path / "out_R2.fq")
    assert not list(tmp_path.glob(".bbnorm-input-*"))
    assert len(calls) == (1 if fail else 2)


def test_single_input_normalization_does_not_prepare_copy(tmp_path):
    with patch("bbmapy.reformat") as reformat, patch(
        "bbmapy.bbnorm", return_value=("", "")
    ) as normalize:
        create_sample_file(
            file_path="interleaved.fq.gz", output_file=str(tmp_path / "out.fq"),
            subset_type="bbnorm", sample_size=17, interleaved=True,
        )
    reformat.assert_not_called()
    assert normalize.call_args.kwargs["interleaved"] == "t"


def test_native_bbnorm_variable_length_mates(tmp_path, monkeypatch):
    """Exercise more than one reader buffer and both native BBNorm passes."""
    import bbmapy
    from bbmapy.base import find_bbtools_path

    tools = find_bbtools_path()
    if not tools or not shutil.which("java"):
        pytest.skip("Native BBTools and Java are required")
    rng = random.Random(493)
    genome = "".join(rng.choices("ACGT", k=5000))
    inputs = [tmp_path / f"reads_R{mate}.fq.gz" for mate in (1, 2)]
    records = {}
    for mate, path in enumerate(inputs, 1):
        with gzip.open(path, "wt") as stream:
            for i in range(4000):
                length = (95 if mate == 1 else 175) + i % 19
                start = rng.randrange(len(genome) - length)
                seq = genome[start:start + length]
                header = f"pair{i}/{mate}"
                records[header] = seq
                stream.write(f"@{header}\n{seq}\n+\n{'I' * length}\n")

    logs = []

    def run(script, **kwargs):
        kwargs.pop("capture_output", None)
        if script == "bbnorm.sh":
            kwargs.update(bits=8, cells=1000000, prefilter="f")
        command = [str(Path(tools) / script), "-Xmx256m"]
        command.extend(f"{key}={value}" for key, value in kwargs.items())
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        if script == "bbnorm.sh":
            logs.append(result.stderr)
        return result.stdout, result.stderr

    monkeypatch.setattr(bbmapy, "reformat", lambda **kw: run("reformat.sh", **kw))
    monkeypatch.setattr(bbmapy, "bbnorm", lambda **kw: run("bbnorm.sh", **kw))
    outputs = [tmp_path / f"out_R{mate}.fq" for mate in (1, 2)]
    create_sample_file(
        file_path=",".join(map(str, inputs)),
        output_file=",".join(map(str, outputs)),
        subset_type="bbnorm", sample_size=10000, bbnorm_min_depth=1,
    )
    counts = re.findall(r"Total reads in:\s+(\d+)", logs[0])
    assert counts == ["8000", "8000"]
    mates = []
    for path in outputs:
        lines = path.read_text().splitlines()
        assert len(lines) == 4000 * 4
        names = []
        for i in range(0, len(lines), 4):
            name = lines[i][1:]
            assert lines[i + 1] == records[name]
            names.append(name.rsplit("/", 1)[0])
        mates.append(names)
    assert mates[0] == mates[1]
    assert not list(tmp_path.glob(".bbnorm-input-*"))
