# Champ web viewer

Sandbox for the Champ fight (Ironclad A20) on the held-out bench2k decks: play / edit the board and compare the
PV net (value, priors), the teacher's live search (guided rollout, 20k sims, 8 particles) and teacher playouts.

    cmake -S . -B build/champ-viewer -DCMAKE_BUILD_TYPE=Release && cmake --build build/champ-viewer --target champ_session -j4
    .venv/bin/python gui/champ_server.py          # http://127.0.0.1:8733/champ/   (?edit opens edit mode)

- `apps/champ_viewer/session.cpp`: stateless battle rebuild per request (start + ops: act / set / move / add / remove).
- `gui/champ_server.py`: bench starts + teacher fights, PV session (D5 model), ≤2 concurrent search/playout processes.
- `gui/web/champ/`: the page. Finished unedited bench fights → `user_fights.jsonl` (rebuilt-verified); tally in the header.

Page state (deck patch, ops, filter, edit mode, N / sims, auto, replay position, undos) lives in the URL hash;
finished searches / playouts are cached server-side in `cache.jsonl` (gitignored), so resumed links show them instantly.

Measured costs: teacher search 0.4–0.6 s/decision (1 core); one 20k-sim playout ≈ 11–16 s (n=4), so N=10 ≈ 65 s on 2 cores.
