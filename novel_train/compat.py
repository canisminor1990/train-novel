"""Call TRL constructors across versions that renamed arguments."""

from __future__ import annotations

import re

_ALIASES = {
    "eval_strategy": "evaluation_strategy",
    "max_seq_length": "max_length",
}


def call_accepting(factory, kwargs: dict):
    pending = dict(kwargs)
    renamed: set[str] = set()
    while True:
        try:
            return factory(**pending)
        except TypeError as exc:
            match = re.search(r"unexpected keyword argument '(\w+)'", str(exc))
            if not match:
                raise
            name = match.group(1)
            if name not in pending:
                raise
            alias = _ALIASES.get(name)
            if alias and alias not in pending and name not in renamed:
                pending[alias] = pending.pop(name)
                renamed.add(name)
                continue
            pending.pop(name)
            if not pending:
                raise
