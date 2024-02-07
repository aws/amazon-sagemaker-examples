from __future__ import print_function, unicode_literals

import six

from prompt_toolkit.application import Application
from prompt_toolkit.eventloop import get_event_loop
from prompt_toolkit.formatted_text import FormattedText, to_formatted_text
from prompt_toolkit.input import DummyInput
from prompt_toolkit.layout import Layout
from prompt_toolkit.output import ColorDepth, Output
from prompt_toolkit.output.defaults import create_output, get_default_output
from prompt_toolkit.renderer import \
    print_formatted_text as renderer_print_formatted_text
from prompt_toolkit.styles import (
    BaseStyle,
    default_pygments_style,
    default_ui_style,
    merge_styles,
)

__all__ = [
    'print_formatted_text',
    'print_container',
    'clear',
    'set_title',
    'clear_title',
]


def print_formatted_text(*values, **kwargs):
    """
    ::

        print_formatted_text(*values, sep=' ', end='\\n', file=None, flush=False, style=None, output=None)

    Print text to stdout. This is supposed to be compatible with Python's print
    function, but supports printing of formatted text. You can pass a
    :class:`~prompt_toolkit.formatted_text.FormattedText`,
    :class:`~prompt_toolkit.formatted_text.HTML` or
    :class:`~prompt_toolkit.formatted_text.ANSI` object to print formatted
    text.

    * Print HTML as follows::

        print_formatted_text(HTML('<i>Some italic text</i> <ansired>This is red!</ansired>'))

        style = Style.from_dict({
            'hello': '#ff0066',
            'world': '#884444 italic',
        })
        print_formatted_text(HTML('<hello>Hello</hello> <world>world</world>!'), style=style)

    * Print a list of (style_str, text) tuples in the given style to the
      output.  E.g.::

        style = Style.from_dict({
            'hello': '#ff0066',
            'world': '#884444 italic',
        })
        fragments = FormattedText([
            ('class:hello', 'Hello'),
            ('class:world', 'World'),
        ])
        print_formatted_text(fragments, style=style)

    If you want to print a list of Pygments tokens, wrap it in
    :class:`~prompt_toolkit.formatted_text.PygmentsTokens` to do the
    conversion.

    :param values: Any kind of printable object, or formatted string.
    :param sep: String inserted between values, default a space.
    :param end: String appended after the last value, default a newline.
    :param style: :class:`.Style` instance for the color scheme.
    :param include_default_pygments_style: `bool`. Include the default Pygments
        style when set to `True` (the default).
    """
    # Pop kwargs (Python 2 compatibility).
    sep = kwargs.pop('sep', ' ')
    end = kwargs.pop('end', '\n')
    file = kwargs.pop('file', None)
    flush = kwargs.pop('flush', False)
    style = kwargs.pop('style', None)
    output = kwargs.pop('output', None)
    color_depth = kwargs.pop('color_depth', None)
    style_transformation = kwargs.pop('style_transformation', None)
    include_default_pygments_style = kwargs.pop('include_default_pygments_style', True)
    assert not kwargs
    assert not (output and file)
    assert style is None or isinstance(style, BaseStyle)

    # Build/merge style.
    styles = [default_ui_style()]
    if include_default_pygments_style:
        styles.append(default_pygments_style())
    if style:
        styles.append(style)

    merged_style = merge_styles(styles)

    # Create Output object.
    if output is None:
        if file:
            output = create_output(stdout=file)
        else:
            output = get_default_output()

    assert isinstance(output, Output)

    # Get color depth.
    color_depth = color_depth or ColorDepth.default()

    # Merges values.
    def to_text(val):
        # Normal lists which are not instances of `FormattedText` are
        # considered plain text.
        if isinstance(val, list) and not isinstance(val, FormattedText):
            return to_formatted_text('{0}'.format(val))
        return to_formatted_text(val, auto_convert=True)

    fragments = []
    for i, value in enumerate(values):
        fragments.extend(to_text(value))

        if sep and i != len(values) - 1:
            fragments.extend(to_text(sep))

    fragments.extend(to_text(end))

    # Print output.
    renderer_print_formatted_text(
        output, fragments, merged_style, color_depth=color_depth,
        style_transformation=style_transformation)

    # Flush the output stream.
    if flush:
        output.flush()


def print_container(container, file=None):
    """
    Print any layout to the output in a non-interactive way.

    Example usage::

        from prompt_toolkit.widgets import Frame, TextArea
        print_container(
            Frame(TextArea(text='Hello world!')))
    """
    if file:
        output = create_output(stdout=file)
    else:
        output = get_default_output()

    def exit_immediately():
        # Use `call_from_executor` to exit "soon", so that we still render one
        # initial time, before exiting the application.
        get_event_loop().call_from_executor(
             lambda: app.exit())

    app = Application(
        layout=Layout(container=container),
        output=output,
        input=DummyInput())
    app.run(pre_run=exit_immediately)


def clear():
    """
    Clear the screen.
    """
    out = get_default_output()
    out.erase_screen()
    out.cursor_goto(0, 0)
    out.flush()


def set_title(text):
    """
    Set the terminal title.
    """
    assert isinstance(text, six.text_type)

    output = get_default_output()
    output.set_title(text)


def clear_title():
    """
    Erase the current title.
    """
    set_title('')
