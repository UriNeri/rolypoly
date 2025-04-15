import os
import re
from pathlib import Path


def update_version():
    """Update version in pyproject.toml, README.md, and meta.yaml"""
    # Update pyproject.toml
    pyproject_path = Path("pyproject.toml")
    # meta_yaml_path = Path("recipes/bbmapy/meta.yaml")
    readme_path = Path("README.md")

    if pyproject_path.exists():
        content = pyproject_path.read_text()
        # Update version in pyproject.toml
        version_match = re.search(r'version = "0.0.(\d+)"', content)
        new_version = int(version_match.group(1)) + 1
        new_version = f"0.0.{new_version}"
        content = re.sub(r'version = "[^"]+"', f'version = "{new_version}"', content)
        pyproject_path.write_text(content)
        print(f"Updated pyproject.toml version to {new_version}")

    # Update README.md
    if readme_path.exists():
        content = readme_path.read_text()
        # Update version badge or version section in README
        content = re.sub(
            r"Current BBMap version: 0.0.\d+",
            f"Current BBMap version: {new_version}",
            content,
        )
        readme_path.write_text(content)
        print(f"Updated README.md version to {new_version}")


def main():
    # Get package root directory
    package_root = Path(__file__).parent.parent
    print(package_root)
    vendor_dir = package_root / "bbmapy/vendor"
    os.makedirs(vendor_dir, exist_ok=True)

    # Get and update version
    update_version()
    print("Update completed successfully!")


if __name__ == "__main__":
    main()
    # this scripts is for development purposes only, when creating a new release to pypi and maybe bioconda.
