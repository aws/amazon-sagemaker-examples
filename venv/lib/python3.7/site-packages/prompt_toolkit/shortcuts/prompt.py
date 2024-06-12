"""
Line editing functionality.
---------------------------

This provides a UI for a line input, similar to GNU Readline, libedit and
linenoise.

Either call the `prompt` function for every line input. Or create an instance
of the :class:`.PromptSession` class and call the `prompt` method from that
class. In the second case, we'll have a 'session' that keeps all the state like
the history in between several calls.

There is a lot of overlap between the arguments taken by the `prompt` function
and the `PromptSession` (like `completer`, `style`, etcetera). There we have
the freedom to decide which settings we want for the whole 'session', and which
we want for an individual `prompt`.

Example::

        # Simple `prompt` call.
        result = prompt('Say something: ')

        # Using a 'session'.
        s = PromptSession()
        result = s.prompt('Say something: ')
"""
from __future__ import unicode_literals

import contextlib
import threading
import time
from functools import partial

from six import text_type

from prompt_toolkit.application import Application
from prompt_toolkit.application.current import get_app
from prompt_toolkit.auto_suggest import DynamicAutoSuggest
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.clipboard import DynamicClipboard, InMemoryClipboard
from prompt_toolkit.completion import DynamicCompleter, ThreadedCompleter
from prompt_toolkit.document import Document
from prompt_toolkit.enums import DEFAULT_BUFFER, SEARCH_BUFFER, EditingMode
from prompt_toolkit.eventloop import (
    From,
    Return,
    ensure_future,
    get_event_loop,
)
from prompt_toolkit.filters import (
    Condition,
    has_arg,
    has_focus,
    is_done,
    is_true,
    renderer_height_is_known,
    to_filter,
)
from prompt_toolkit.formatted_text import (
    fragment_list_to_text,
    merge_formatted_text,
    to_formatted_text,
)
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.input.defaults import get_default_input
from prompt_toolkit.key_binding.bindings.auto_suggest import (
    load_auto_suggest_bindings,
)
from prompt_toolkit.key_binding.bindings.completion import (
    display_completions_like_readline,
)
from prompt_toolkit.key_binding.bindings.open_in_editor import (
    load_open_in_editor_bindings,
)
from prompt_toolkit.key_binding.key_bindings import (
    ConditionalKeyBindings,
    DynamicKeyBindings,
    KeyBindings,
    KeyBindingsBase,
    merge_key_bindings,
)
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import Float, FloatContainer, HSplit, Window
from prompt_toolkit.layout.containers import ConditionalContainer, WindowAlign
from prompt_toolkit.layout.controls import (
    BufferControl,
    FormattedTextControl,
    SearchBufferControl,
)
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.layout.menus import (
    CompletionsMenu,
    MultiColumnCompletionsMenu,
)
from prompt_toolkit.layout.processors import (
    AppendAutoSuggestion,
    ConditionalProcessor,
    DisplayMultipleCursors,
    DynamicProcessor,
    HighlightIncrementalSearchProcessor,
    HighlightSelectionProcessor,
    PasswordProcessor,
    ReverseSearchProcessor,
    merge_processors,
)
from prompt_toolkit.layout.utils import explode_text_fragments
from prompt_toolkit.lexers import DynamicLexer
from prompt_toolkit.output.defaults import get_default_output
from prompt_toolkit.styles import (
    BaseStyle,
    ConditionalStyleTransformation,
    DynamicStyle,
    DynamicStyleTransformation,
    StyleTransformation,
    SwapLightAndDarkStyleTransformation,
    merge_style_transformations,
)
from prompt_toolkit.utils import get_cwidth, suspend_to_background_supported
from prompt_toolkit.validation import DynamicValidator
from prompt_toolkit.widgets.toolbars import (
    SearchToolbar,
    SystemToolbar,
    ValidationToolbar,
)

__all__ = [
    'PromptSession',
    'prompt',
    'confirm',
    'create_confirm_session',  # Used by '_display_completions_like_readline'.
    'CompleteStyle',
]


def _split_multiline_prompt(get_prompt_text):
    """
    Take a `get_prompt_text` function and return three new functions instead.
    One that tells whether this prompt consists of multiple lines; one that
    returns the fragments to be shown on the lines above the input; and another
    one with the fragments to be shown at the first line of the input.
    """
    def has_before_fragments():
        for fragment, char in get_prompt_text():
            if '\n' in char:
                return True
        return False

    def before():
        result = []
        found_nl = False
        for fragment, char in reversed(explode_text_fragments(get_prompt_text())):
            if found_nl:
                result.insert(0, (fragment, char))
            elif char == '\n':
                found_nl = True
        return result

    def first_input_line():
        result = []
        for fragment, char in reversed(explode_text_fragments(get_prompt_text())):
            if char == '\n':
                break
            else:
                result.insert(0, (fragment, char))
        return result

    return has_before_fragments, before, first_input_line


class _RPrompt(Window):
    " The prompt that is displayed on the right side of the Window. "
    def __init__(self, get_formatted_text):
        super(_RPrompt, self).__init__(
            FormattedTextControl(get_formatted_text),
            align=WindowAlign.RIGHT,
            style='class:rprompt')


class CompleteStyle:
    " How to display autocompletions for the prompt. "
    COLUMN = 'COLUMN'
    MULTI_COLUMN = 'MULTI_COLUMN'
    READLINE_LIKE = 'READLINE_LIKE'


class PromptSession(object):
    """
    PromptSession for a prompt application, which can be used as a GNU Readline
    replacement.

    This is a wrapper around a lot of ``prompt_toolkit`` functionality and can
    be a replacement for `raw_input`.

    All parameters that expect "formatted text" can take either just plain text
    (a unicode object), a list of ``(style_str, text)`` tuples or an HTML object.

    Example usage::

        s = PromptSession(message='>')
        text = s.prompt()

    :param message: Plain text or formatted text to be shown before the prompt.
        This can also be a callable that returns formatted text.
    :param multiline: `bool` or :class:`~prompt_toolkit.filters.Filter`.
        When True, prefer a layout that is more adapted for multiline input.
        Text after newlines is automatically indented, and search/arg input is
        shown below the input, instead of replacing the prompt.
    :param wrap_lines: `bool` or :class:`~prompt_toolkit.filters.Filter`.
        When True (the default), automatically wrap long lines instead of
        scrolling horizontally.
    :param is_password: Show asterisks instead of the actual typed characters.
    :param editing_mode: ``EditingMode.VI`` or ``EditingMode.EMACS``.
    :param vi_mode: `bool`, if True, Identical to ``editing_mode=EditingMode.VI``.
    :param complete_while_typing: `bool` or
        :class:`~prompt_toolkit.filters.Filter`. Enable autocompletion while
        typing.
    :param validate_while_typing: `bool` or
        :class:`~prompt_toolkit.filters.Filter`. Enable input validation while
        typing.
    :param enable_history_search: `bool` or
        :class:`~prompt_toolkit.filters.Filter`. Enable up-arrow parting
        string matching.
    :param search_ignore_case:
        :class:`~prompt_toolkit.filters.Filter`. Search case insensitive.
    :param lexer: :class:`~prompt_toolkit.lexers.Lexer` to be used for the
        syntax highlighting.
    :param validator: :class:`~prompt_toolkit.validation.Validator` instance
        for input validation.
    :param completer: :class:`~prompt_toolkit.completion.Completer` instance
        for input completion.
    :param complete_in_thread: `bool` or
        :class:`~prompt_toolkit.filters.Filter`. Run the completer code in a
        background thread in order to avoid blocking the user interface.
        For ``CompleteStyle.READLINE_LIKE``, this setting has no effect. There
        we always run the completions in the main thread.
    :param reserve_space_for_menu: Space to be reserved for displaying the menu.
        (0 means that no space needs to be reserved.)
    :param auto_suggest: :class:`~prompt_toolkit.auto_suggest.AutoSuggest`
        instance for input suggestions.
    :param style: :class:`.Style` instance for the color scheme.
    :param include_default_pygments_style: `bool` or
        :class:`~prompt_toolkit.filters.Filter`. Tell whether the default
        styling for Pygments lexers has to be included. By default, this is
        true, but it is recommended to be disabled if another Pygments style is
        passed as the `style` argument, otherwise, two Pygments styles will be
        merged.
    :param style_transformation:
        :class:`~prompt_toolkit.style.StyleTransformation` instance.
    :param swap_light_and_dark_colors: `bool` or
        :class:`~prompt_toolkit.filters.Filter`. When enabled, apply
        :class:`~prompt_toolkit.style.SwapLightAndDarkStyleTransformation`.
        This is useful for switching between dark and light terminal
        backgrounds.
    :param enable_system_prompt: `bool` or
        :class:`~prompt_toolkit.filters.Filter`. Pressing Meta+'!' will show
        a system prompt.
    :param enable_suspend: `bool` or :class:`~prompt_toolkit.filters.Filter`.
        Enable Control-Z style suspension.
    :param enable_open_in_editor: `bool` or
        :class:`~prompt_toolkit.filters.Filter`. Pressing 'v' in Vi mode or
        C-X C-E in emacs mode will open an external editor.
    :param history: :class:`~prompt_toolkit.history.History` instance.
    :param clipboard: :class:`~prompt_toolkit.clipboard.Clipboard` instance.
        (e.g. :class:`~prompt_toolkit.clipboard.InMemoryClipboard`)
    :param rprompt: Text or formatted text to be displayed on the right side.
        This can also be a callable that returns (formatted) text.
    :param bottom_toolbar: Formatted text or callable which is supposed to
        return formatted text.
    :param prompt_continuation: Text that needs to be displayed for a multiline
        prompt continuation. This can either be formatted text or a callable
        that takes a `prompt_width`, `line_number` and `wrap_count` as input
        and returns formatted text.
    :param complete_style: ``CompleteStyle.COLUMN``,
        ``CompleteStyle.MULTI_COLUMN`` or ``CompleteStyle.READLINE_LIKE``.
    :param mouse_support: `bool` or :class:`~prompt_toolkit.filters.Filter`
        to enable mouse support.
    :param refresh_interval: (number; in seconds) When given, refresh the UI
        every so many seconds.
    :param inputhook: None or an Inputhook callable that takes an
        `InputHookContext` object.
    """
    _fields = (
        'message', 'lexer', 'completer', 'complete_in_thread', 'is_password',
        'editing_mode', 'key_bindings', 'is_password', 'bottom_toolbar',
        'style', 'style_transformation', 'swap_light_and_dark_colors',
        'color_depth', 'include_default_pygments_style', 'rprompt',
        'multiline', 'prompt_continuation', 'wrap_lines',
        'enable_history_search', 'search_ignore_case', 'complete_while_typing',
        'validate_while_typing', 'complete_style', 'mouse_support',
        'auto_suggest', 'clipboard', 'validator', 'refresh_interval',
        'input_processors', 'enable_system_prompt', 'enable_suspend',
        'enable_open_in_editor', 'reserve_space_for_menu', 'tempfile_suffix',
        'inputhook')

    def __init__(
            self,
            message='',
            multiline=False,
            wrap_lines=True,
            is_password=False,
            vi_mode=False,
            editing_mode=EditingMode.EMACS,
            complete_while_typing=True,
            validate_while_typing=True,
            enable_history_search=False,
            search_ignore_case=False,
            lexer=None,
            enable_system_prompt=False,
            enable_suspend=False,
            enable_open_in_editor=False,
            validator=None,
            completer=None,
            complete_in_thread=False,
            reserve_space_for_menu=8,
            complete_style=None,
            auto_suggest=None,
            style=None,
            style_transformation=None,
            swap_light_and_dark_colors=False,
            color_depth=None,
            include_default_pygments_style=True,
            history=None,
            clipboard=None,
            prompt_continuation=None,
            rprompt=None,
            bottom_toolbar=None,
            mouse_support=False,
            input_processors=None,
            key_bindings=None,
            erase_when_done=False,
            tempfile_suffix='.txt',
            inputhook=None,

            refresh_interval=0,
            input=None,
            output=None):
        assert style is None or isinstance(style, BaseStyle)
        assert style_transformation is None or isinstance(style_transformation, StyleTransformation)
        assert input_processors is None or isinstance(input_processors, list)
        assert key_bindings is None or isinstance(key_bindings, KeyBindingsBase)

        # Defaults.
        output = output or get_default_output()
        input = input or get_default_input()

        history = history or InMemoryHistory()
        clipboard = clipboard or InMemoryClipboard()

        # Ensure backwards-compatibility, when `vi_mode` is passed.
        if vi_mode:
            editing_mode = EditingMode.VI

        # Store all settings in this class.
        self.input = input
        self.output = output

        # Store all settings in this class.
        for name in self._fields:
            if name not in ('editing_mode', ):
                value = locals()[name]
                setattr(self, name, value)

        # Create buffers, layout and Application.
        self.history = history
        self.default_buffer = self._create_default_buffer()
        self.search_buffer = self._create_search_buffer()
        self.layout = self._create_layout()
        self.app = self._create_application(editing_mode, erase_when_done)

    def _dyncond(self, attr_name):
        """
        Dynamically take this setting from this 'PromptSession' class.
        `attr_name` represents an attribute name of this class. Its value
        can either be a boolean or a `Filter`.

        This returns something that can be used as either a `Filter`
        or `Filter`.
        """
        @Condition
        def dynamic():
            value = getattr(self, attr_name)
            return to_filter(value)()
        return dynamic

    def _create_default_buffer(self):
        """
        Create and return the default input buffer.
        """
        dyncond = self._dyncond

        # Create buffers list.
        def accept(buff):
            """ Accept the content of the default buffer. This is called when
            the validation succeeds. """
            self.app.exit(result=buff.document.text)
            return True  # Keep text, we call 'reset' later on.

        return Buffer(
            name=DEFAULT_BUFFER,
                # Make sure that complete_while_typing is disabled when
                # enable_history_search is enabled. (First convert to Filter,
                # to avoid doing bitwise operations on bool objects.)
            complete_while_typing=Condition(lambda:
                is_true(self.complete_while_typing) and not
                is_true(self.enable_history_search) and not
                self.complete_style == CompleteStyle.READLINE_LIKE),
            validate_while_typing=dyncond('validate_while_typing'),
            enable_history_search=dyncond('enable_history_search'),
            validator=DynamicValidator(lambda: self.validator),
            completer=DynamicCompleter(lambda:
                ThreadedCompleter(self.completer)
                if self.complete_in_thread and self.completer
                else self.completer),
            history=self.history,
            auto_suggest=DynamicAutoSuggest(lambda: self.auto_suggest),
            accept_handler=accept,
            tempfile_suffix=lambda: self.tempfile_suffix)

    def _create_search_buffer(self):
        return Buffer(name=SEARCH_BUFFER)

    def _create_layout(self):
        """
        Create `Layout` for this prompt.
        """
        dyncond = self._dyncond

        # Create functions that will dynamically split the prompt. (If we have
        # a multiline prompt.)
        has_before_fragments, get_prompt_text_1, get_prompt_text_2 = \
            _split_multiline_prompt(self._get_prompt)

        default_buffer = self.default_buffer
        search_buffer = self.search_buffer

        # Create processors list.
        all_input_processors = [
            HighlightIncrementalSearchProcessor(),
            HighlightSelectionProcessor(),
            ConditionalProcessor(AppendAutoSuggestion(),
                                 has_focus(default_buffer) & ~is_done),
            ConditionalProcessor(PasswordProcessor(), dyncond('is_password')),
            DisplayMultipleCursors(),

            # Users can insert processors here.
            DynamicProcessor(lambda: merge_processors(self.input_processors or [])),
        ]

        # Create bottom toolbars.
        bottom_toolbar = ConditionalContainer(
            Window(FormattedTextControl(
                        lambda: self.bottom_toolbar,
                        style='class:bottom-toolbar.text'),
                   style='class:bottom-toolbar',
                   dont_extend_height=True,
                   height=Dimension(min=1)),
            filter=~is_done & renderer_height_is_known &
                    Condition(lambda: self.bottom_toolbar is not None))

        search_toolbar = SearchToolbar(
            search_buffer,
            ignore_case=dyncond('search_ignore_case'))

        search_buffer_control = SearchBufferControl(
            buffer=search_buffer,
            input_processors=[
                ReverseSearchProcessor(),
            ],
            ignore_case=dyncond('search_ignore_case'))

        system_toolbar = SystemToolbar(
            enable_global_bindings=dyncond('enable_system_prompt'))

        def get_search_buffer_control():
            " Return the UIControl to be focused when searching start. "
            if is_true(self.multiline):
                return search_toolbar.control
            else:
                return search_buffer_control

        default_buffer_control = BufferControl(
            buffer=default_buffer,
            search_buffer_control=get_search_buffer_control,
            input_processors=all_input_processors,
            include_default_input_processors=False,
            lexer=DynamicLexer(lambda: self.lexer),
            preview_search=True)

        default_buffer_window = Window(
            default_buffer_control,
            height=self._get_default_buffer_control_height,
            get_line_prefix=partial(
                self._get_line_prefix, get_prompt_text_2=get_prompt_text_2),
            wrap_lines=dyncond('wrap_lines'))

        @Condition
        def multi_column_complete_style():
            return self.complete_style == CompleteStyle.MULTI_COLUMN

        # Build the layout.
        layout = HSplit([
            # The main input, with completion menus floating on top of it.
            FloatContainer(
                HSplit([
                    ConditionalContainer(
                        Window(
                            FormattedTextControl(get_prompt_text_1),
                            dont_extend_height=True),
                        Condition(has_before_fragments)
                    ),
                    ConditionalContainer(
                        default_buffer_window,
                        Condition(lambda:
                            get_app().layout.current_control != search_buffer_control),
                    ),
                    ConditionalContainer(
                        Window(search_buffer_control),
                        Condition(lambda:
                            get_app().layout.current_control == search_buffer_control),
                    ),
                ]),
                [
                    # Completion menus.
                    # NOTE: Especially the multi-column menu needs to be
                    #       transparent, because the shape is not always
                    #       rectangular due to the meta-text below the menu.
                    Float(xcursor=True,
                          ycursor=True,
                          transparent=True,
                          content=CompletionsMenu(
                              max_height=16,
                              scroll_offset=1,
                              extra_filter=has_focus(default_buffer) &
                                  ~multi_column_complete_style)),
                    Float(xcursor=True,
                          ycursor=True,
                          transparent=True,
                          content=MultiColumnCompletionsMenu(
                              show_meta=True,
                              extra_filter=has_focus(default_buffer) &
                                  multi_column_complete_style)),
                    # The right prompt.
                    Float(right=0, top=0, hide_when_covering_content=True,
                          content=_RPrompt(lambda: self.rprompt)),
                ]
            ),
            ConditionalContainer(
                ValidationToolbar(),
                filter=~is_done),
            ConditionalContainer(
                system_toolbar,
                dyncond('enable_system_prompt') & ~is_done),

            # In multiline mode, we use two toolbars for 'arg' and 'search'.
            ConditionalContainer(
                Window(FormattedTextControl(self._get_arg_text), height=1),
                dyncond('multiline') & has_arg),
            ConditionalContainer(search_toolbar, dyncond('multiline') & ~is_done),
            bottom_toolbar,
        ])

        return Layout(layout, default_buffer_window)

    def _create_application(self, editing_mode, erase_when_done):
        """
        Create the `Application` object.
        """
        dyncond = self._dyncond

        # Default key bindings.
        auto_suggest_bindings = load_auto_suggest_bindings()
        open_in_editor_bindings = load_open_in_editor_bindings()
        prompt_bindings = self._create_prompt_bindings()

        # Create application
        application = Application(
            layout=self.layout,
            style=DynamicStyle(lambda: self.style),
            style_transformation=merge_style_transformations([
                DynamicStyleTransformation(lambda: self.style_transformation),
                ConditionalStyleTransformation(
                    SwapLightAndDarkStyleTransformation(),
                    dyncond('swap_light_and_dark_colors'),
                ),
            ]),
            include_default_pygments_style=dyncond('include_default_pygments_style'),
            clipboard=DynamicClipboard(lambda: self.clipboard),
            key_bindings=merge_key_bindings([
                merge_key_bindings([
                    auto_suggest_bindings,
                    ConditionalKeyBindings(open_in_editor_bindings,
                        dyncond('enable_open_in_editor') &
                        has_focus(DEFAULT_BUFFER)),
                    prompt_bindings
                ]),
                DynamicKeyBindings(lambda: self.key_bindings),
            ]),
            mouse_support=dyncond('mouse_support'),
            editing_mode=editing_mode,
            erase_when_done=erase_when_done,
            reverse_vi_search_direction=True,
            color_depth=lambda: self.color_depth,

            # I/O.
            input=self.input,
            output=self.output)

        # During render time, make sure that we focus the right search control
        # (if we are searching). - This could be useful if people make the
        # 'multiline' property dynamic.
        '''
        def on_render(app):
            multiline = is_true(self.multiline)
            current_control = app.layout.current_control

            if multiline:
                if current_control == search_buffer_control:
                    app.layout.current_control = search_toolbar.control
                    app.invalidate()
            else:
                if current_control == search_toolbar.control:
                    app.layout.current_control = search_buffer_control
                    app.invalidate()

        app.on_render += on_render
        '''

        return application

    def _create_prompt_bindings(self):
        """
        Create the KeyBindings for a prompt application.
        """
        kb = KeyBindings()
        handle = kb.add
        default_focused = has_focus(DEFAULT_BUFFER)

        @Condition
        def do_accept():
            return (not is_true(self.multiline) and
                    self.app.layout.has_focus(DEFAULT_BUFFER))

        @handle('enter', filter=do_accept & default_focused)
        def _(event):
            " Accept input when enter has been pressed. "
            self.default_buffer.validate_and_handle()

        @Condition
        def readline_complete_style():
            return self.complete_style == CompleteStyle.READLINE_LIKE

        @handle('tab', filter=readline_complete_style & default_focused)
        def _(event):
            " Display completions (like Readline). "
            display_completions_like_readline(event)

        @handle('c-c', filter=default_focused)
        def _(event):
            " Abort when Control-C has been pressed. "
            event.app.exit(exception=KeyboardInterrupt, style='class:aborting')

        @Condition
        def ctrl_d_condition():
            """ Ctrl-D binding is only active when the default buffer is selected
            and empty. """
            app = get_app()
            return (app.current_buffer.name == DEFAULT_BUFFER and
                    not app.current_buffer.text)

        @handle('c-d', filter=ctrl_d_condition & default_focused)
        def _(event):
            " Exit when Control-D has been pressed. "
            event.app.exit(exception=EOFError, style='class:exiting')

        suspend_supported = Condition(suspend_to_background_supported)

        @Condition
        def enable_suspend():
            return to_filter(self.enable_suspend)()

        @handle('c-z', filter=suspend_supported & enable_suspend)
        def _(event):
            """
            Suspend process to background.
            """
            event.app.suspend_to_background()

        return kb

    @contextlib.contextmanager
    def _auto_refresh_context(self):
        " Return a context manager for the auto-refresh loop. "
        done = [False]  # nonlocal

        # Enter.

        def run():
            while not done[0]:
                time.sleep(self.refresh_interval)
                self.app.invalidate()

        if self.refresh_interval:
            t = threading.Thread(target=run)
            t.daemon = True
            t.start()

        try:
            yield
        finally:
            # Exit.
            done[0] = True

    def prompt(
            self,
            # When any of these arguments are passed, this value is overwritten
            # in this PromptSession.
            message=None,  # `message` should go first, because people call it
                           # as positional argument.
            editing_mode=None, refresh_interval=None, vi_mode=None, lexer=None,
            completer=None, complete_in_thread=None, is_password=None,
            key_bindings=None, bottom_toolbar=None, style=None,
            color_depth=None, include_default_pygments_style=None,
            style_transformation=None, swap_light_and_dark_colors=None,
            rprompt=None, multiline=None, prompt_continuation=None,
            wrap_lines=None, enable_history_search=None,
            search_ignore_case=None, complete_while_typing=None,
            validate_while_typing=None, complete_style=None, auto_suggest=None,
            validator=None, clipboard=None, mouse_support=None,
            input_processors=None, reserve_space_for_menu=None,
            enable_system_prompt=None, enable_suspend=None,
            enable_open_in_editor=None, tempfile_suffix=None, inputhook=None,

            # Following arguments are specific to the current `prompt()` call.
            async_=False, default='', accept_default=False, pre_run=None):
        """
        Display the prompt. All the arguments are a subset of the
        :class:`~.PromptSession` class itself.

        This will raise ``KeyboardInterrupt`` when control-c has been pressed
        (for abort) and ``EOFError`` when control-d has been pressed (for
        exit).

        Additional arguments, specific for this prompt:

        :param async_: When `True` return a `Future` instead of waiting for the
            prompt to finish.
        :param default: The default input text to be shown. (This can be edited
            by the user).
        :param accept_default: When `True`, automatically accept the default
            value without allowing the user to edit the input.
        :param pre_run: Callable, called at the start of `Application.run`.
        """
        assert isinstance(default, text_type)

        # NOTE: We used to create a backup of the PromptSession attributes and
        #       restore them after exiting the prompt. This code has been
        #       removed, because it was confusing and didn't really serve a use
        #       case. (People were changing `Application.editing_mode`
        #       dynamically and surprised that it was reset after every call.)

        # Take settings from 'prompt'-arguments.
        for name in self._fields:
            value = locals()[name]
            if value is not None:
                setattr(self, name, value)

        if vi_mode:
            self.editing_mode = EditingMode.VI

        def pre_run2():
            if pre_run:
                pre_run()

            if accept_default:
                # Validate and handle input. We use `call_from_executor` in
                # order to run it "soon" (during the next iteration of the
                # event loop), instead of right now. Otherwise, it won't
                # display the default value.
                get_event_loop().call_from_executor(
                    self.default_buffer.validate_and_handle)

        def run_sync():
            with self._auto_refresh_context():
                self.default_buffer.reset(Document(default))
                return self.app.run(inputhook=self.inputhook, pre_run=pre_run2)

        def run_async():
            with self._auto_refresh_context():
                self.default_buffer.reset(Document(default))
                result = yield From(self.app.run_async(pre_run=pre_run2))
                raise Return(result)

        if async_:
            return ensure_future(run_async())
        else:
            return run_sync()

    @property
    def editing_mode(self):
        return self.app.editing_mode

    @editing_mode.setter
    def editing_mode(self, value):
        self.app.editing_mode = value

    def _get_default_buffer_control_height(self):
        # If there is an autocompletion menu to be shown, make sure that our
        # layout has at least a minimal height in order to display it.
        if (self.completer is not None and
                self.complete_style != CompleteStyle.READLINE_LIKE):
            space = self.reserve_space_for_menu
        else:
            space = 0

        if space and not get_app().is_done:
            buff = self.default_buffer

            # Reserve the space, either when there are completions, or when
            # `complete_while_typing` is true and we expect completions very
            # soon.
            if buff.complete_while_typing() or buff.complete_state is not None:
                return Dimension(min=space)

        return Dimension()

    def _get_prompt(self):
        return to_formatted_text(self.message, style='class:prompt')

    def _get_continuation(self, width, line_number, wrap_count):
        """
        Insert the prompt continuation.

        :param width: The width that was used for the prompt. (more or less can
            be used.)
        :param line_number:
        :param wrap_count: Amount of times that the line has been wrapped.
        """
        prompt_continuation = self.prompt_continuation

        if callable(prompt_continuation):
            prompt_continuation = prompt_continuation(width, line_number, wrap_count)

        # When the continuation prompt is not given, choose the same width as
        # the actual prompt.
        if not prompt_continuation and is_true(self.multiline):
            prompt_continuation = ' ' * width

        return to_formatted_text(
            prompt_continuation, style='class:prompt-continuation')

    def _get_line_prefix(self, line_number, wrap_count, get_prompt_text_2):
        """
        Return whatever needs to be inserted before every line.
        (the prompt, or a line continuation.)
        """
        # First line: display the "arg" or the prompt.
        if line_number == 0 and wrap_count == 0:
            if not is_true(self.multiline) and get_app().key_processor.arg is not None:
                return self._inline_arg()
            else:
                return get_prompt_text_2()

        # For the next lines, display the appropriate continuation.
        prompt_width = get_cwidth(fragment_list_to_text(get_prompt_text_2()))
        return self._get_continuation(prompt_width, line_number, wrap_count)

    def _get_arg_text(self):
        " 'arg' toolbar, for in multiline mode. "
        arg = self.app.key_processor.arg
        if arg == '-':
            arg = '-1'

        return [
            ('class:arg-toolbar', 'Repeat: '),
            ('class:arg-toolbar.text', arg)
        ]

    def _inline_arg(self):
        " 'arg' prefix, for in single line mode. "
        app = get_app()
        if app.key_processor.arg is None:
            return []
        else:
            arg = app.key_processor.arg

            return [
                ('class:prompt.arg', '(arg: '),
                ('class:prompt.arg.text', str(arg)),
                ('class:prompt.arg', ') '),
            ]


def prompt(*a, **kw):
    """ The global `prompt` function. This will create a new `PromptSession`
    instance for every call.  """
    # Input and output arguments have to be passed to the 'PromptSession'
    # class, not its method.
    input = kw.pop('input', None)
    output = kw.pop('output', None)
    history = kw.pop('history', None)

    session = PromptSession(input=input, output=output, history=history)
    return session.prompt(*a, **kw)


prompt.__doc__ = PromptSession.prompt.__doc__


def create_confirm_session(message, suffix=' (y/n) '):
    """
    Create a `PromptSession` object for the 'confirm' function.
    """
    assert isinstance(message, text_type)
    bindings = KeyBindings()

    @bindings.add('y')
    @bindings.add('Y')
    def yes(event):
        session.default_buffer.text = 'y'
        event.app.exit(result=True)

    @bindings.add('n')
    @bindings.add('N')
    def no(event):
        session.default_buffer.text = 'n'
        event.app.exit(result=False)

    @bindings.add(Keys.Any)
    def _(event):
        " Disallow inserting other text. "
        pass

    complete_message = merge_formatted_text([message, suffix])
    session = PromptSession(complete_message, key_bindings=bindings)
    return session


def confirm(message='Confirm?', suffix=' (y/n) '):
    """
    Display a confirmation prompt that returns True/False.
    """
    session = create_confirm_session(message, suffix)
    return session.prompt()
