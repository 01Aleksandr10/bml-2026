import io
import os
import sys


class Tee:
    """Дублирует stdout в файл с результатами работы скрипта."""

    def __init__(self, script_path):
        self.script_path = script_path
        self.buf = io.StringIO()

    def write(self, s):
        sys.__stdout__.write(s)
        self.buf.write(s)

    def flush(self):
        sys.__stdout__.flush()

    def save(self, filename=None):
        sys.stdout = sys.__stdout__
        if filename is None:
            base = os.path.splitext(os.path.basename(self.script_path))[0]
            filename = base.split("_")[0] + "_output.txt"
        path = os.path.join(os.path.dirname(os.path.abspath(self.script_path)), filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.buf.getvalue())


def start_capture():
    t = Tee(sys.argv[0])
    sys.stdout = t
    return t
