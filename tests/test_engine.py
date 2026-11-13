"""The engine fingerprint ignores cosmetic edits and catches logic edits."""

from kinema.engine import _normalized


def test_comments_docstrings_and_formatting_do_not_change_the_hash():
    a = 'def f(x):\n    """Doc."""\n    return x + 1  # add one\n'
    b = "def f( x ):\n    # reworded comment\n    '''A different docstring.'''\n\n    return x+1\n"
    assert _normalized(a) == _normalized(b)


def test_logic_and_message_changes_do():
    base = "def f(x):\n    return x + 1\n"
    assert _normalized(base) != _normalized("def f(x):\n    return x + 2\n")
    msg = 'def f():\n    return "flagged"\n'
    assert _normalized(msg) != _normalized('def f():\n    return "flagged!"\n')


def test_docstring_only_function_still_parses():
    assert _normalized('def f():\n    """Only a docstring."""\n') == _normalized("def f():\n    pass\n")
