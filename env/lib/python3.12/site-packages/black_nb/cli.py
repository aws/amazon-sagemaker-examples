"""Apply Black to Jupyter notebooks."""

# Original work Copyright © 2018-2020 Łukasz Langa
# Modified work Copyright © 2019-2020 Tom Catling, Liam Coatman

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.


from pathlib import Path
from typing import Any, Dict, List, Optional, Pattern, Tuple

import black
from black.files import find_project_root
import click
import nbformat
from attr import dataclass

DEFAULT_LINE_LENGTH = black.DEFAULT_LINE_LENGTH
DEFAULT_INCLUDES = r"\.ipynb$"
DEFAULT_EXCLUDES = (
    rf"{black.DEFAULT_EXCLUDES.rstrip(')/')}|\.ipynb_checkpoints)/"
)
TARGET_VERSIONS = {
    black.TargetVersion.PY36,
    black.TargetVersion.PY37,
    black.TargetVersion.PY38,
}


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "-l",
    "--line-length",
    type=int,
    default=DEFAULT_LINE_LENGTH,
    help="How many characters per line to allow.",
    show_default=True,
)
@click.option(
    "--check",
    is_flag=True,
    help=(
        "Don't write the files back, just return the status.  Return code 0 "
        "means nothing would change.  Return code 1 means some files would be "
        "reformatted.  Return code 123 means there was an internal error."
    ),
)
@click.option(
    "--include",
    type=str,
    default=DEFAULT_INCLUDES,
    callback=black.validate_regex,
    help=(
        "A regular expression that matches files and directories that should "
        "be included on recursive searches.  An empty value means all files "
        "are included regardless of the name.  Use forward slashes for "
        "directories on all platforms (Windows, too).  Exclusions are "
        "calculated first, inclusions later."
    ),
    show_default=True,
)
@click.option(
    "--exclude",
    type=str,
    default=DEFAULT_EXCLUDES,
    callback=black.validate_regex,
    help=(
        "A regular expression that matches files and directories that should "
        "be excluded on recursive searches. An empty value means no paths are "
        "excluded. Use forward slashes for directories on all platforms "
        "(Windows, too). Exclusions are calculated first, inclusions later."
    ),
    show_default=True,
)
@click.option(
    "--extend-exclude",
    type=str,
    callback=black.validate_regex,
    help=(
        "Like --exclude, but adds additional files and directories on top of "
        "the excluded ones. (Useful if you simply want to add to the default)"
    ),
)
@click.option(
    "--force-exclude",
    type=str,
    callback=black.validate_regex,
    help=(
        "Like --exclude, but files and directories matching this regex will "
        "be excluded even when they are passed explicitly as arguments."
    ),
)
@click.option(
    "--stdin-filename",
    type=str,
    help=(
        "The name of the file when passing it through stdin. Useful to make "
        "sure Black will respect --force-exclude option on some "
        "editors that rely on using stdin."
    ),
)
@click.option(
    "-q",
    "--quiet",
    is_flag=True,
    help=(
        "Don't emit non-error messages to stderr. Errors are still emitted, "
        "silence those with 2>/dev/null."
    ),
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help=(
        "Also emit messages to stderr about files that were not changed or "
        "were ignored due to --exclude=."
    ),
)
@click.option(
    "-o",
    "--clear-output",
    is_flag=True,
    help="Clear cell output as part of formatting.",
)
@click.argument(
    "src",
    nargs=-1,
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=True,
        readable=True,
        allow_dash=True,
    ),
    is_eager=True,
)
@click.option(
    "--config",
    type=click.Path(
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=True,
        allow_dash=False,
    ),
    is_eager=True,
    callback=black.read_pyproject_toml,
    help="Read configuration from PATH.",
)
@click.pass_context
def cli(
    ctx: click.Context,
    line_length: int,
    check: bool,
    include: Pattern[str],
    exclude: Pattern[str],
    extend_exclude: Optional[Pattern[str]],
    force_exclude: Optional[Pattern[str]],
    stdin_filename: Optional[str],
    quiet: bool,
    verbose: bool,
    clear_output: bool,
    src: Tuple[str, ...],
    config: Optional[str],
) -> None:
    """
    The uncompromising code formatter, for Jupyter notebooks.
    """
    write_back = black.WriteBack.from_configuration(check=check, diff=False)
    mode = black.Mode(
        target_versions=TARGET_VERSIONS,
        line_length=line_length,
        is_pyi=False,
        string_normalization=True,
    )

    if config and verbose:
        black.out(f"Using configuration from {config}.", bold=False, fg="blue")

    report = black.Report(check=check, quiet=quiet, verbose=verbose)

    ctx.ensure_object(dict)
    root, method = find_project_root(src)
    ctx.obj["root"] = root

    sources = black.get_sources(
        ctx=ctx,
        src=src,
        quiet=quiet,
        verbose=verbose,
        include=include,
        exclude=exclude,
        force_exclude=force_exclude,
        report=report,
        extend_exclude=extend_exclude,
        stdin_filename=stdin_filename,
    )

    black.path_empty(
        sources,
        "No Jupyter notebooks are present to be formatted. Nothing to do 😴",
        quiet,
        verbose,
        ctx,
    )

    for source in sources:
        reformat_one(
            src=source,
            write_back=write_back,
            mode=mode,
            clear_output=clear_output,
            report=report,
            quiet=quiet,
            verbose=verbose,
        )

    if verbose or not quiet:
        black.out("All done! ✨ 🍰 ✨")
        click.secho(str(report), err=True)
    ctx.exit(report.return_code)


def reformat_one(
    src: Path,
    write_back: black.WriteBack,
    mode: black.FileMode,
    clear_output: bool,
    report: black.Report,
    quiet: bool,
    verbose: bool,
) -> None:
    """Reformat a single file under `src`."""
    try:

        sub_report = SubReport(write_back=write_back)
        changed = black.Changed.NO

        cache: black.Cache = {}
        if write_back is not black.WriteBack.DIFF:
            cache = black.read_cache(mode)
            res_src = src.resolve()
            if res_src in cache and cache[res_src] == black.get_cache_info(
                res_src
            ):
                changed = black.Changed.CACHED
        if changed is not black.Changed.CACHED:
            sub_report = format_file_in_place(
                src,
                write_back=write_back,
                mode=mode,
                clear_output=clear_output,
                sub_report=sub_report,
            )
            if sub_report.change_count or sub_report.output_change_count:
                changed = black.Changed.YES
        if (
            write_back is black.WriteBack.YES
            and changed is not black.Changed.CACHED
        ) or (
            write_back is black.WriteBack.CHECK and changed is black.Changed.NO
        ):
            black.write_cache(cache, [src], mode)
        report.done(src, changed)
        if changed is not black.Changed.CACHED and (verbose or not quiet):
            click.secho(f"    {sub_report}", err=True)
    except Exception as exc:
        report.failed(src, str(exc))


def format_file_in_place(
    src: Path,
    write_back: black.WriteBack,
    mode: black.FileMode,
    clear_output: bool,
    sub_report: "SubReport",
) -> "SubReport":
    """
    Format file under `src` path. Return True if changed.
    If `write_back` is YES, write reformatted code to the file.
    """
    try:
        src_contents = nbformat.read(
            str(src),
            as_version=nbformat.NO_CONVERT,
        )
    except nbformat.reader.NotJSONError:
        raise black.InvalidInput("Not JSON")
    except AttributeError:
        raise black.InvalidInput("No cells")

    dst_cells: List[Dict[Any, Any]] = []
    for cell in src_contents["cells"]:
        if cell["cell_type"] == "code":
            try:
                cell["source"] = format_cell_source(cell["source"], mode=mode)
                sub_report.done(black.Changed.YES)
            except black.NothingChanged:
                sub_report.done(black.Changed.NO)
            except black.InvalidInput:
                sub_report.failed()
            if clear_output:
                try:
                    (
                        cell["outputs"],
                        cell["execution_count"],
                    ) = clear_cell_outputs(
                        cell["outputs"], cell["execution_count"]
                    )
                    sub_report.done_output(black.Changed.YES)
                except black.NothingChanged:
                    sub_report.done_output(black.Changed.NO)
        dst_cells.append(cell)
    src_contents["cells"] = dst_cells

    if write_back is black.WriteBack.YES:
        nbformat.write(src_contents, str(src))

    return sub_report


def clear_cell_outputs(
    src_outputs: List[str], src_execution_count: int
) -> Tuple[List[str], None]:
    if src_outputs == [] and src_execution_count is None:
        raise black.NothingChanged
    return [], None


def format_cell_source(
    src_contents: str, *, mode: black.FileMode
) -> black.FileContent:
    """
    Reformat contents of cell and return new contents.
    Additionally confirm that the reformatted code is valid by calling
    :func:`assert_equivalent` and :func:`assert_stable` on it.
    """

    if src_contents.strip() == "":
        raise black.NothingChanged

    dst_contents = format_str(src_contents, mode=mode)

    if src_contents == dst_contents:
        raise black.NothingChanged

    assert_equivalent(src_contents, dst_contents)
    assert_stable(dst_contents, mode=mode)

    return dst_contents


def format_str(
    src_contents: str,
    *,
    mode: black.FileMode = black.FileMode(),
) -> black.FileContent:

    # Strip trailing semicolon because Black removes it, but it is an
    # important feature in notebooks.
    # Only a single trailing semicolon is supported. If the cell contains
    # multiple trailing semicolons black_nb will fail.
    trailing_semi_colon = src_contents.rstrip()[-1] == ";"

    src_contents = hide_magic(src_contents)
    dst_contents = black.format_str(src_contents, mode=mode)
    dst_contents = dst_contents.rstrip()

    # Replace the missing semi colon, except when Black didn't remove it
    # which happens if the last line is a comment
    if trailing_semi_colon and dst_contents.rstrip()[-1] != ";":
        dst_contents = f"{dst_contents};"

    dst_contents = reveal_magic(dst_contents)
    return dst_contents


def assert_equivalent(src: str, dst: str) -> None:
    black.assert_equivalent(hide_magic(src), hide_magic(dst))


def assert_stable(
    dst: str,
    mode: black.FileMode = black.FileMode(),
) -> None:
    new_dst = format_str(dst, mode=mode)
    if dst != new_dst:
        raise AssertionError(
            "INTERNAL ERROR: Black produced different code on the second pass "
            "of the formatter."
        ) from None


def contains_magic(line: str) -> bool:
    if len(line) == 0:
        return False
    else:
        return (
            line[0] == "%"
            or line[0] == "!"
            or (line[-1] == "?" and line.lstrip()[0] != "#")
        )


def hide_magic(source: str) -> str:
    """
    Black can't deal with cell or line magic, so we
    disguise it as a comment. This keeps it in the same
    place in the reformatted code.
    """

    def _hide_magic_line(line: str) -> str:
        return f"###MAGIC###{line}" if contains_magic(line) else line

    return "\n".join(_hide_magic_line(line) for line in source.split("\n"))


def reveal_magic(source: str) -> str:
    """
    Reveal any notebook magic hidden by hide_magic().
    """
    return source.replace("###MAGIC###", "")


@dataclass
class SubReport:
    """
    Provides a reformatting counter for notebook cells.
    Can be rendered with `str(report)`.
    """

    write_back: black.WriteBack
    change_count: int = 0
    same_count: int = 0
    failure_count: int = 0
    output_change_count: int = 0
    output_same_count: int = 0

    def done(self, changed: black.Changed) -> None:
        """
        Increment the counter for successful reformatting.
        """
        if changed is black.Changed.YES:
            self.change_count += 1
        else:
            self.same_count += 1

    def done_output(self, changed: black.Changed) -> None:
        """
        Increment the counter for successful clear output.
        """
        if changed is black.Changed.YES:
            self.output_change_count += 1
        else:
            self.output_same_count += 1

    def failed(self) -> None:
        """
        Increment the counter for failed reformatting.
        """
        self.failure_count += 1

    def __str__(self) -> str:
        """
        Render a report of the current state.
        """
        if self.write_back is black.WriteBack.CHECK:
            reformatted = "would be reformatted"
            unchanged = "would be left unchanged"
            failed = "would fail to reformat"
            cleared = "would be cleared"
        else:
            reformatted = "reformatted"
            unchanged = "left unchanged"
            failed = "failed to reformat"
            cleared = "cleared"
        report = []
        if self.change_count:
            s = "s" if self.change_count > 1 else ""
            report.append(
                click.style(
                    f"{self.change_count} cell{s} {reformatted}", bold=True
                )
            )
        if self.same_count:
            s = "s" if self.same_count > 1 else ""
            report.append(f"{self.same_count} cell{s} {unchanged}")
        if self.failure_count:
            s = "s" if self.failure_count > 1 else ""
            report.append(
                click.style(f"{self.failure_count} cell{s} {failed}", fg="red")
            )
        if self.output_change_count:
            s = "s" if self.change_count > 1 else ""
            report.append(
                click.style(
                    f"{self.output_change_count} output{s} {cleared}",
                    bold=True,
                )
            )
        if self.output_same_count:
            s = "s" if self.same_count > 1 else ""
            report.append(f"{self.output_same_count} output{s} {unchanged}")
        return ", ".join(report) + "."


if __name__ == "__main__":
    cli()
