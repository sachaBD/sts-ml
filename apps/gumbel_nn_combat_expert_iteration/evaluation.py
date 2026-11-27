"""Batch compatible evaluation fights; keep each set's published artifacts independent."""
import json
from itertools import zip_longest


def groups(sets):
    """Search settings are TOML tables; equality is independent of key order."""
    batches = []
    for evaluation in sets:
        for batch in batches:
            if batch[0].search == evaluation.search:
                batch.append(evaluation)
                break
        else:
            batches.append([evaluation])
    return batches


class Batch:
    def __init__(self, sets, fights):
        self.fights = []
        self.routes = {}
        # Interleave to keep every set feeding the same queue, even if the last set is small.
        sets = [(s.name, fights[s.name]) for s in sets]
        for row in zip_longest(*(f for _, f in sets)):
            for (name, _), fight in zip(sets, row):
                if fight is None:
                    continue
                key = f"{fight['fight_id']}:batch-{len(self.fights)}"
                self.routes[key] = name, fight['fight_id']
                self.fights.append({**fight, 'fight_id': key})

    def split(self, results):
        if {r['fight_id'] for r in results} != self.routes.keys() or len(results) != len(self.routes):
            raise ValueError('evaluation batch did not return exactly one result per fight')
        by_set = {}
        for row in results:  # nn_play emits completion order, not input order
            name, fight_id = self.routes[row['fight_id']]
            by_set.setdefault(name, []).append({**row, 'fight_id': fight_id})
        return by_set


def publish(directory, fights, results, samples, *, retain_samples):
    """Publish an ordinary per-set results journal with per-set byte offsets."""
    directory.mkdir(parents=True)
    (directory / 'starts.jsonl').write_text(''.join(json.dumps(f) + '\n' for f in fights))
    # nn_play writes one sample file; read spans without loading the batch's replay into RAM.
    offset = 0
    if retain_samples:
        with samples.open('rb') as source, (directory / samples.name).open('wb') as dest:
            for row in results:
                span = row['sample_range']
                source.seek(span['begin'])
                remaining = span['end'] - span['begin']
                while remaining > 0:
                    data = source.read(min(remaining, 1024 * 1024))
                    if not data:
                        raise ValueError(f'{samples}: truncated evaluation samples')
                    dest.write(data)
                    remaining -= len(data)
    for row in results:
        if span := row.get('sample_range'):
            end = offset + span['end'] - span['begin']
            row['sample_range'] = {**span, 'begin': offset, 'end': end}
            offset = end
    (directory / 'results.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in results))
