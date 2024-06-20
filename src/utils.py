from __future__ import annotations

import os


def create_dirs(dirs: list) -> None:
    for d in dirs:
        if not os.path.isdir(d):
            os.mkdir(d)
