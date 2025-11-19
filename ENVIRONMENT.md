# Python Environment

This project uses a virtual environment located at `.venv`.

Create the venv (if not already created):

```bash
python3 -m venv .venv
```

Activate it (bash/zsh):

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

Run Python with the venv's interpreter directly:

```bash
.venv/bin/python your_script.py
```

If you use an editor, point its Python interpreter to `.venv/bin/python`.
