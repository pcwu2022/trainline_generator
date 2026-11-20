"""Prepare California and Texas inputs from official Census state polygons."""
import argparse
from pathlib import Path
from urllib.request import urlopen
from tools.prepare_usa_northeast import SOURCE, prepare_states


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--states-kml', type=Path)
    args = parser.parse_args()
    raw = args.states_kml.read_bytes() if args.states_kml else urlopen(SOURCE, timeout=60).read()
    for name, state in [('california', 'CA'), ('texas', 'TX')]:
        prepare_states(raw, Path('data') / name, {state})


if __name__ == '__main__':
    main()
