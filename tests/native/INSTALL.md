# To apply this suite

This is a validated staging copy of the deliverable `tests/native/` suite.

```bash
mkdir -p tests/native
cp "/tmp/genesis/task_c0eca8a3abe0896f"/{__init__.py,CONTEXT.md,test_codec.py,test_native_loader.py} tests/native/
source scripts/dev-env.sh && .venv/bin/python -m pytest tests/native -q
```

Expected: `46 passed` (validated from this staging directory on the same host).
