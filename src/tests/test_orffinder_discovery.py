"""NCBI executable discovery, installation and failure handling."""
import gzip
import io
import os
from pathlib import Path
import subprocess
import sys

import pytest
from rolypoly.utils import various


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(various, '_ORFFINDER_PATH', None)
    monkeypatch.setattr(various.shutil, 'which', lambda name: None)
    monkeypatch.delenv('PIXI_ENVIRONMENT_PREFIX', raising=False)
    monkeypatch.delenv('CONDA_PREFIX', raising=False)
    monkeypatch.setattr(sys, 'prefix', str(tmp_path / 'interpreter'))
    import platform
    monkeypatch.setattr(platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(platform, 'machine', lambda: 'x86_64')
    return tmp_path


def executable(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('#!/bin/sh\necho "ORFfinder: 0.test"\n')
    path.chmod(0o755)
    return path


def test_path_precedes_environment(isolated, monkeypatch):
    path = executable(isolated / 'path tool' / 'ORFfinder')
    monkeypatch.setattr(various.shutil, 'which', lambda name: str(path))
    monkeypatch.setenv('CONDA_PREFIX', str(isolated / 'unused'))
    assert various.ensure_orffinder(allow_download=False) == path


@pytest.mark.parametrize('variable', ['PIXI_ENVIRONMENT_PREFIX', 'CONDA_PREFIX'])
def test_environment_not_on_path(isolated, monkeypatch, variable):
    path = executable(isolated / 'env with spaces' / 'bin' / 'ORFfinder')
    monkeypatch.setenv(variable, str(path.parent.parent))
    assert various.ensure_orffinder(allow_download=False) == path


def test_download_is_atomic_and_executable(isolated, monkeypatch):
    import urllib.request
    prefix = isolated / 'new env'
    monkeypatch.setenv('PIXI_ENVIRONMENT_PREFIX', str(prefix))
    payload = gzip.compress(b'#!/bin/sh\necho "ORFfinder: 0.test"\n')
    monkeypatch.setattr(urllib.request, 'urlopen', lambda *a, **k: io.BytesIO(payload))
    path = various.ensure_orffinder()
    assert path == prefix / 'bin' / 'ORFfinder'
    assert os.access(path, os.X_OK)
    assert not list(path.parent.glob('ORFfinder-download-*'))
    assert various.ensure_orffinder(allow_download=False) == path


def test_noexec_environment_uses_private_temp(isolated, monkeypatch):
    import urllib.request
    prefix = isolated / 'noexec'
    monkeypatch.setenv('CONDA_PREFIX', str(prefix))
    monkeypatch.setattr(urllib.request, 'urlopen', lambda *a, **k: io.BytesIO(
        gzip.compress(b'#!/bin/sh\necho "ORFfinder: 0.test"\n')))
    original_run = subprocess.run
    def run(command, **kwargs):
        if str(command[0]).startswith(str(prefix)):
            raise PermissionError('noexec mount')
        return original_run(command, **kwargs)
    monkeypatch.setattr(subprocess, 'run', run)
    path = various.ensure_orffinder(temp_dir=isolated)
    assert path.parent.parent == isolated
    assert not (prefix / 'bin' / 'ORFfinder').exists()
    assert various.ensure_orffinder(allow_download=False) == path


def test_unsupported_platform_does_not_download(isolated, monkeypatch):
    import platform
    import urllib.request
    monkeypatch.setattr(platform, 'machine', lambda: 'aarch64')
    monkeypatch.setattr(urllib.request, 'urlopen', lambda *a, **k: pytest.fail('download'))
    with pytest.raises(RuntimeError, match='Linux/aarch64'):
        various.ensure_orffinder()


def test_failed_download_leaves_no_partial_binary(isolated, monkeypatch):
    import urllib.request
    monkeypatch.setenv('CONDA_PREFIX', str(isolated / 'env'))
    monkeypatch.setattr(urllib.request, 'urlopen', lambda *a, **k: io.BytesIO(b'not gzip'))
    with pytest.raises(RuntimeError, match='Unable to install'):
        various.ensure_orffinder(temp_dir=isolated)
    assert not list(isolated.rglob('ORFfinder'))
    assert not list(isolated.rglob('ORFfinder-download-*'))


def test_prediction_passes_paths_as_arguments(isolated, monkeypatch):
    from rolypoly.utils.bio.translation import predict_orfs_orffinder
    calls = []
    monkeypatch.setattr(subprocess, 'run', lambda command, **kwargs: calls.append(command))
    predict_orfs_orffinder('input with spaces.fa', 'output with spaces.faa', 90, 1,
                          executable=isolated / 'ORFfinder')
    assert calls[0][calls[0].index('-in') + 1] == 'input with spaces.fa'
    assert calls[0][calls[0].index('-out') + 1] == 'output with spaces.faa'
