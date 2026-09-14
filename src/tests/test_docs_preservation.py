import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('docs_export', Path(__file__).parents[1] / 'setup/export_command_help_to_docs.py')
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


def test_refresh_preserves_manual_content_verbatim():
    before = '# Report\n\nCustom description with Unicode: α.\n\n'
    after = '\n\n## Caveats\nKeep this!\n\n## Known bugs\nDo not lose.\n'
    old = before + export.OPTIONS_BEGIN + '\n## Options\nold\n' + export.OPTIONS_END + after
    new = export.update_cli_options(old, '- `--new`: current option')
    assert new == before + export.OPTIONS_BEGIN + '\n## Options\n\n- `--new`: current option\n' + export.OPTIONS_END + after
    assert export.update_cli_options(new, '- `--new`: current option') == new


def test_migration_preserves_existing_text_and_is_idempotent():
    original = '# Report\n\n## Options\n\n- existing option\n\n## Caveats\nManual notes\n'
    migrated = export.update_cli_options(original, 'ignored', migrate=True)
    assert migrated.replace(export.OPTIONS_BEGIN+'\n', '').replace('\n'+export.OPTIONS_END, '') == original
    assert export.update_cli_options(migrated, 'ignored', migrate=True) == migrated


@pytest.mark.parametrize('text', ['# Unmarked', export.OPTIONS_BEGIN, export.OPTIONS_END+export.OPTIONS_BEGIN, export.OPTIONS_BEGIN*2+export.OPTIONS_END])
def test_unmarked_or_invalid_pages_rejected(text):
    with pytest.raises(ValueError):
        export.update_cli_options(text, 'new options')


def test_custom_template_has_markers():
    template = (Path(__file__).parents[1] / 'setup/help_export_template.md').read_text()
    for content in (template, export.DEFAULT_TEMPLATE):
        assert content.count(export.OPTIONS_BEGIN) == content.count(export.OPTIONS_END) == 1


def test_batch_validation_prevents_partial_writes(tmp_path, monkeypatch):
    import click
    first = tmp_path / 'first.md'
    second = tmp_path / 'second.md'
    original = export.OPTIONS_BEGIN + '\nold\n' + export.OPTIONS_END
    first.write_text(original)
    second.write_text('# Handwritten page\n')
    monkeypatch.setattr(export, 'get_click_command', lambda name: click.Command(name))
    monkeypatch.setattr(export, 'get_command_summary', lambda name: 'Summary')
    monkeypatch.setattr(export, 'get_command_description', lambda name: 'Description')
    with pytest.raises(ValueError, match='Unmarked'):
        export.ensure_docs_pages(tmp_path, tmp_path/'scaffolds', export.DEFAULT_TEMPLATE,
            ['first', 'second'], {'first': first, 'second': second}, True, False)
    assert first.read_text() == original
    assert second.read_text() == '# Handwritten page\n'
