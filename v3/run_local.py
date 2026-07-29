"""run_local.py -- drive the confirmatory run in bounded time slices.

The runner in protocol.py is resumable (it appends one JSON line per completed
unit and skips units already present), so the whole study can be driven by
repeated short invocations. This wrapper simply stops cleanly after a wall
clock budget so it can be called in a loop.
"""
import signal
import sys

import protocol as PR

DS1 = "/sessions/gifted-kind-shannon/mnt/Salwa article 19 July/EEG/dataset"
DS2A = ("/sessions/gifted-kind-shannon/mnt/Salwa article 19 July/"
        "BCI Competition IV Dataset 2a")


class Budget(BaseException):
    """Deliberately NOT an Exception: protocol.run() wraps each unit in
    `except Exception`, so a plain Exception here would be swallowed and
    logged as a spurious unit failure instead of stopping the slice."""


def _alarm(sig, frm):
    raise Budget()


if __name__ == "__main__":
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 38
    dsets = (sys.argv[2].split(",") if len(sys.argv) > 2
             else ["ds2a_binary", "ds2a_4class", "ds1"])
    out = sys.argv[3] if len(sys.argv) > 3 else "/tmp/v3run"
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(budget)
    cfg = PR.RunCfg(ds1_dir=DS1, ds2a_dir=DS2A, out_dir=out,
                    datasets=tuple(dsets), run_eegnet=False)
    try:
        PR.run(cfg)
        print("ALL DONE")
    except Budget:
        print("budget reached; rerun to continue")
