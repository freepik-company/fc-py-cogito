"""Click command classes for displaying root options in nested help."""

import click


class _RootOptionsHelpMixin:
    """Append the root command's options to nested command help."""

    def format_options(
        self, ctx: click.Context, formatter: click.HelpFormatter
    ) -> None:
        super().format_options(ctx, formatter)

        root_ctx = ctx.find_root()
        if root_ctx is ctx:
            return

        help_records = []
        for param in root_ctx.command.get_params(root_ctx):
            if not isinstance(param, click.Option) or param.name == "help":
                continue
            help_record = param.get_help_record(root_ctx)
            if help_record is not None:
                help_records.append(help_record)

        if help_records:
            with formatter.section("Global Options"):
                formatter.write_dl(help_records)


class RootOptionsCommand(_RootOptionsHelpMixin, click.Command):
    """Command that includes applicable root options in its help."""


class RootOptionsGroup(_RootOptionsHelpMixin, click.Group):
    """Group that includes applicable root options in its help."""

    command_class = RootOptionsCommand
