#!/usr/bin/env python3
"""
Automated code linting script.
Runs pylint on the codebase and fixes common issues.
"""

import os
import subprocess
import sys
from typing import List, Tuple


def find_python_files(directory: str) -> List[str]:
    """Find all Python files in the given directory and subdirectories."""
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files


def run_pylint(files: List[str]) -> Tuple[int, str]:
    """Run pylint on the given files."""
    try:
        cmd = ['pylint', '--rcfile=.pylintrc'] + files
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False)
        return result.returncode, result.stdout
    except subprocess.CalledProcessError as e:
        return e.returncode, e.output


def fix_common_issues(file_path: str):
    """Fix common code style issues in the given file."""
    try:
        # Run autopep8 for basic PEP 8 fixes
        subprocess.run(['autopep8', '--in-place',
                       '--aggressive', file_path], check=True)

        # Run isort to sort imports
        subprocess.run(['isort', file_path], check=True)

        print(f"✓ Fixed common issues in {file_path}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error fixing {file_path}: {e}")


def main():
    """Main function to run the linting process."""
    # Get Python files in the current directory
    python_files = find_python_files('.')

    if not python_files:
        print("No Python files found.")
        return

    print(f"Found {len(python_files)} Python files to check.")

    # Run pylint
    print("\nRunning pylint...")
    return_code, output = run_pylint(python_files)

    # Print pylint output
    print("\nPylint Results:")
    print("=" * 80)
    print(output)
    print("=" * 80)

    # Fix common issues if pylint found problems
    if return_code != 0:
        print("\nAttempting to fix common issues...")
        for file in python_files:
            fix_common_issues(file)

        # Run pylint again to check results
        print("\nRe-running pylint after fixes...")
        new_return_code, new_output = run_pylint(python_files)

        print("\nUpdated Pylint Results:")
        print("=" * 80)
        print(new_output)
        print("=" * 80)

        if new_return_code == 0:
            print("\n✓ All issues fixed!")
        else:
            print("\n! Some issues remain. Manual review may be needed.")
    else:
        print("\n✓ No issues found!")


if __name__ == '__main__':
    main()
