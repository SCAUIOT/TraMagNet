"""argparse compat: ``BooleanOptionalAction`` (``--flag`` / ``--no-flag``)."""

from __future__ import annotations

import argparse

try:
    BooleanOptionalAction = argparse.BooleanOptionalAction
except AttributeError:

    class BooleanOptionalAction(argparse.Action):
        def __init__(
            self,
            option_strings,
            dest,
            default=None,
            type=None,
            choices=None,
            required=False,
            help=None,
            metavar=None,
        ):
            if type is not None or choices is not None:
                raise ValueError("BooleanOptionalAction does not support type or choices")
            if metavar is not None:
                raise ValueError("BooleanOptionalAction does not support metavar")
            if required:
                raise ValueError("BooleanOptionalAction is not compatible with required=True")
            if len(option_strings) != 1:
                raise ValueError("BooleanOptionalAction only accepts a single option string")
            opt = option_strings[0]
            if not opt.startswith("--") or "." in opt:
                raise ValueError("BooleanOptionalAction only accepts long options without a dot")
            option_strings = [opt, opt.replace("--", "--no-", 1)]
            super().__init__(
                option_strings=option_strings,
                dest=dest,
                nargs=0,
                const=True,
                default=default,
                type=type,
                choices=choices,
                required=required,
                help=help,
                metavar=metavar,
            )

        def __call__(self, parser, namespace, values, option_string=None):
            if option_string is not None and option_string.startswith("--no-"):
                setattr(namespace, self.dest, False)
            else:
                setattr(namespace, self.dest, True)
