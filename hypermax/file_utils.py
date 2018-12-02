from contextlib import contextmanager
import tempfile
import os

# Windows doesn't support opening a NamedTemporaryFile.
# Solution inspired in https://stackoverflow.com/a/46501017/147507
@contextmanager
def ClosedNamedTempFile(contents):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as f:
            file_name = f.name
            f.write(contents)
        yield file_name
    finally:
        os.unlink(file_name)