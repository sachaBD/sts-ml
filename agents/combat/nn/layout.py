"""An encoding's token layout, read from its layout.toml: the one description its encoder and model share."""
from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class Kind:
    id: int
    features: int


@dataclass(frozen=True)
class Layout:
    path: Path
    name: str
    version: int
    width: int
    id_capacity: int
    link_offset: int
    kinds: dict[str, Kind]
    links: dict[str, int]
    card_zones: dict[str, int]
    action_kinds: dict[str, int]


def load(path: Path) -> Layout:
    raw = tomllib.loads(path.read_text())
    if raw['header'] != ['kind', 'id', 'id2']:
        raise ValueError(f'{path}: header must be kind, id, id2')  # tokens.TokenRows reads these columns
    return Layout(path, raw['name'], raw['version'], raw['width'], raw['id_capacity'], raw['link_offset'],
                  {name: Kind(**kind) for name, kind in raw['kinds'].items()}, raw['links'],
                  raw.get('card_zones', {}), raw.get('action_kinds', {}))
