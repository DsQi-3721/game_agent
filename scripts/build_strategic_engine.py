"""Build isolated poke-engine 0.0.48 bindings with persistent guided search."""
import argparse
from pathlib import Path
import shutil
import subprocess
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--python', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = args.out / 'source'
    shutil.copytree(args.source, source, dirs_exist_ok=True, ignore=shutil.ignore_patterns('target', '.git'))
    native = root / 'external/foulplay/native'
    shutil.copy(native / 'strategic_mcts.rs', source / 'src/strategic_mcts.rs')
    shutil.copy(native / 'strategic_binding.rs', source / 'poke-engine-py/src/strategic_binding.rs')
    path = source / 'src/mcts.rs'
    path.write_text(path.read_text() + '\ninclude!("strategic_mcts.rs");\n')
    path = source / 'Cargo.toml'
    path.write_text(path.read_text().replace('[dependencies]', '[dependencies]\nserde = { version = "1", features = ["derive"] }\nserde_json = "1"'))
    path = source / 'poke-engine-py/Cargo.toml'
    path.write_text(path.read_text().replace('[dependencies]', '[dependencies]\nserde_json = "1"'))
    path = source / 'poke-engine-py/src/lib.rs'
    path.write_text(path.read_text().replace('m.add_class::<PyState>()?;',
                    'm.add_class::<PyState>()?;\n    m.add_class::<PySearchTree>()?;')
                    + '\ninclude!("strategic_binding.rs");\n')
    wheels = args.out / 'wheels'; wheels.mkdir(parents=True, exist_ok=True)
    subprocess.run([args.python, '-m', 'pip', 'wheel', '--no-deps', '-w', str(wheels),
                    '--config-settings', 'build-args=--no-default-features --features poke-engine/gen9,poke-engine/terastallization',
                    str(source)], check=True)
    engine = args.out / 'engine'; engine.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(next(wheels.glob('*.whl'))) as wheel:
        wheel.extractall(engine)
    print(engine)


if __name__ == '__main__':
    main()
