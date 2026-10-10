# Champ web viewer

Replay recorded Ironclad A20 vs The Champ fights from any `runs/schema=combat_v4` run on a timeline (play / pause /
speed / step / turn jumps), with the PV net's value and priors at each step. Play any move or edit the board to take
over from the recording.

    cmake -S . -B build/champ-viewer -DCMAKE_BUILD_TYPE=Release && cmake --build build/champ-viewer --target champ_session -j4
    .venv/bin/python gui/champ_server.py          # http://127.0.0.1:8733/champ/
    # link: /champ/?date=2026-10-04&id=champ-ox-c-r01&part=bench-oracle&fight=<fight_id>

- `apps/champ_viewer/session.cpp`: stateless battle rebuild per request (start + ops: act / set / move / add / remove).
- `gui/champ_server.py`: lists combat_v4 datasets (rescanned per page load) and their Champ fights; one PV session (D5 model).
- `gui/web/champ/`: the page. The fight is in the query string; setup patch / ops / edit mode / speed in the URL hash.
- `gui/test_champ_server.py`: one recorded fight replays to its recorded outcome.

History: until 2026-10-05 this was a sandbox on the bench2k decks with live teacher search / playouts; that was
removed. `cache.jsonl` here is its leftover result cache (gitignored, unused).
