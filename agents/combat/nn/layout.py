"""The token layouts in layout.toml, for Python (LAYOUT) and as a generated C++ header.

  python agents/combat/nn/layout.py OUT.hpp    (run by the build)

Tables become namespaces and integers constants, so LAYOUT.inventory_v1.link.owner here is
layout::inventory_v1::link::owner in C++; a kind with features is a Kind {id, features}.
"""
from pathlib import Path
import sys
import tomllib
from types import SimpleNamespace

SOURCE = Path(__file__).with_name('layout.toml')


def namespace(table):
    return SimpleNamespace(**{k: namespace(v) if isinstance(v, dict) else v for k, v in table.items()})


LAYOUT = namespace(tomllib.loads(SOURCE.read_text()))


def cpp(table, indent=''):
    lines = []
    for key, value in table.items():
        if isinstance(value, dict) and set(value) == {'id', 'features'}:
            lines.append(f'{indent}inline constexpr Kind {key}{{{value["id"]}, {value["features"]}}};')
        elif isinstance(value, dict):
            lines += [f'{indent}namespace {key} {{', *cpp(value, indent + '    '), f'{indent}}}  // namespace {key}']
        else:
            lines.append(f'{indent}inline constexpr int {key} = {value};')
    return lines


if __name__ == '__main__':
    header = ['// Generated from agents/combat/nn/layout.toml by agents/combat/nn/layout.py. Do not edit.',
              '#pragma once', '', 'namespace stsrl::nn::layout {', 'struct Kind {', '    int id, features;', '};',
              *cpp(tomllib.loads(SOURCE.read_text())), '}  // namespace stsrl::nn::layout', '']
    Path(sys.argv[1]).write_text('\n'.join(header))
