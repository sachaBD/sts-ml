"""Combine compatible worker jobs and route replies back to their original groups."""
from itertools import zip_longest


def group_by(items, *, key):
    """Stable grouping by equality, including unhashable settings such as dicts."""
    keys, groups = [], []
    for item in items:
        value = key(item)
        if value not in keys:
            keys.append(value)
            groups.append([])
        groups[keys.index(value)].append(item)
    return groups


class Batch:
    def __init__(self, groups, *, id_field):
        self.jobs = []
        self._id_field = id_field
        self._routes = {}
        # Round-robin avoids queuing whole groups back-to-back before the final drain.
        for row in zip_longest(*groups.values()):
            for group, job in zip(groups, row):
                if job is None:
                    continue
                original = job[id_field]
                key = f'{original}:batch-{len(self.jobs)}'
                self._routes[key] = group, original
                self.jobs.append({**job, id_field: key})

    def split(self, replies):
        field = self._id_field
        if len(replies) != len(self._routes) or {r[field] for r in replies} != self._routes.keys():
            raise ValueError('worker batch did not return exactly one reply per job')
        groups = {}
        for reply in replies:
            group, original = self._routes[reply[field]]
            groups.setdefault(group, []).append({**reply, field: original})
        return groups
