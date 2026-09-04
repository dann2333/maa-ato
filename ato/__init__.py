"""ATO — an Arknights-playing AI.

Layers:
    ato.gamedata    versioned ingestion of public game data (the ground truth for the simulator)
    ato.sim         a deterministic offline battle simulator rebuilt from that data
    ato.agent       search, heuristics and learned policies that decide what to do
    ato.perception  reading live battle state off the emulator screen
    ato.control     driving an Android emulator over ADB
    ato.train       imitation / RL / distillation pipelines
    ato.evolve      the self-improvement loop and its gates
"""

__version__ = "0.1.0"
