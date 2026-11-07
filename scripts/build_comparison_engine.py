"""Build the pinned poke-engine root-action comparison binding."""
import argparse
from pathlib import Path
import subprocess
import sys
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path, help='poke_engine-0.0.48.tar.gz source distribution')
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--python', default=sys.executable, help='target Foul Play environment')
    args = parser.parse_args()
    args.build_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.archive) as archive:
        archive.extractall(args.build_dir, filter='data')
    source = args.build_dir / 'poke_engine-0.0.48'
    patch = Path(__file__).resolve().parents[1] / 'external/foulplay/patches/action_comparison.patch'
    subprocess.run(['patch', '-p1', '-i', str(patch)], cwd=source, check=True)
    subprocess.run([args.python, '-m', 'pip', 'install', '--no-deps', '--force-reinstall',
                    '--config-settings',
                    'build-args=--no-default-features --features poke-engine/gen9,poke-engine/terastallization',
                    str(source)], check=True)


if __name__ == '__main__':
    main()
