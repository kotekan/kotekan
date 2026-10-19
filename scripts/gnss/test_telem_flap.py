#!/usr/bin/env python3
"""telem_flap.py --help states the harness's whole first sentence.

    python3 scripts/gnss/test_telem_flap.py
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    r = subprocess.run(
        [sys.executable, os.path.join(HERE, "telem_flap.py"), "--help"],
        capture_output=True,
        text=True,
    )
    text = " ".join(r.stdout.split())  # argparse rewraps to the terminal width
    ok = r.returncode == 0 and "and reconnects at a fixed rate." in text
    print(
        "  [%s] --help carries the description's whole first sentence"
        % ("PASS" if ok else "FAIL")
    )
    print("OK" if ok else "FAILED (1)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
