"""
The API basically only provides one class. You can create a :class:`Script` and
use its methods.

Additionally you can add a debug function with :func:`set_debug_function`.
Alternatively, if you don't need a custom function and are happy with printing
debug messages to stdout, simply call :func:`set_debug_function` without
arguments.
"""
import os
import sys
import warnings
from functools import wraps

import parso
from parso.python import tree

from jedi._compatibility import force_unicode, cast_path, is_py3
from jedi.parser_utils import get_executable_nodes
from jedi import debug
from jedi import settings
from jedi import cache
from jedi.file_io import KnownContentFileIO
from jedi.api import classes
from jedi.api import interpreter
from jedi.api import helpers
from jedi.api.helpers import validate_line_column
from jedi.api.completion import Completion, search_in_module
from jedi.api.keywords import KeywordName
from jedi.api.environment import InterpreterEnvironment
from jedi.api.project import get_default_project, Project
from jedi.api.errors import parso_to_jedi_errors
from jedi.api import refactoring
from jedi.api.refactoring.extract import extract_function, extract_variable
from jedi.inference import InferenceState
from jedi.inference import imports
from jedi.inference.references import find_references
from jedi.inference.arguments import try_iter_content
from jedi.inference.helpers import infer_call_of_leaf
from jedi.inference.sys_path import transform_path_to_dotted
from jedi.inference.syntax_tree import tree_name_to_values
from jedi.inference.value import ModuleValue
from jedi.inference.base_value import ValueSet
from jedi.inference.value.iterable import unpack_tuple_to_dict
from jedi.inference.gradual.conversion import convert_names, convert_values
from jedi.inference.gradual.utils import load_proper_stub_module
from jedi.inference.utils import to_list

# Jedi uses lots and lots of recursion. By setting this a little bit higher, we
# can remove some "maximum recursion depth" errors.
sys.setrecursionlimit(3000)


def _no_python2_support(func):
    # TODO remove when removing Python 2/3.5
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self._inference_state.grammar.version_info < (3, 6) or sys.version_info < (3, 6):
            raise NotImplementedError(
                "No support for refactorings/search on Python 2/3.5"
            )
        return func(self, *args, **kwargs)
    return wrapper


class Script(object):
    """
    A Script is the base for completions, goto or whatever you want to do with
    Jedi. The counter part of this class is :class:`Interpreter`, which works
    with actual dictionaries and can work with a REPL. This class
    should be used when a user edits code in an editor.

    You can either use the ``code`` parameter or ``path`` to read a file.
    Usually you're going to want to use both of them (in an editor).

    The Script's ``sys.path`` is very customizable:

    - If `project` is provided with a ``sys_path``, that is going to be used.
    - If `environment` is provided, its ``sys.path`` will be used
      (see :func:`Environment.get_sys_path <jedi.api.environment.Environment.get_sys_path>`);
    - Otherwise ``sys.path`` will match that of the default environment of
      Jedi, which typically matches the sys path that was used at the time
      when Jedi was imported.

    Most methods have a ``line`` and a ``column`` parameter. Lines in Jedi are
    always 1-based and columns are always zero based. To avoid repetition they
    are not always documented. You can omit both line and column. Jedi will
    then just do whatever action you are calling at the end of the file. If you
    provide only the line, just will complete at the end of that line.

    .. warning:: By default :attr:`jedi.settings.fast_parser` is enabled, which means
        that parso reuses modules (i.e. they are not immutable). With this setting
        Jedi is **not thread safe** and it is also not safe to use multiple
        :class:`.Script` instances and its definitions at the same time.

        If you are a normal plugin developer this should not be an issue. It is
        an issue for people that do more complex stuff with Jedi.

        This is purely a performance optimization and works pretty well for all
        typical usages, however consider to turn the setting off if it causes
        you problems. See also
        `this discussion <https://github.com/davidhalter/jedi/issues/1240>`_.

    :param code: The source code of the current file, separated by newlines.
    :type code: str
    :param line: Deprecated, please use it directly on e.g. ``.complete``
    :type line: int
    :param column: Deprecated, please use it directly on e.g. ``.complete``
    :type column: int
    :param path: The path of the file in the file system, or ``''`` if
        it hasn't been saved yet.
    :type path: str or None
    :param encoding: Deprecated, cast to unicode yourself. The encoding of
        ``code``, if it is not a ``unicode`` object (default ``'utf-8'``).
    :type encoding: str
    :param sys_path: Deprecated, use the project parameter.
    :type sys_path: typing.List[str]
    :param Environment environment: Provide a predefined :ref:`Environment <environments>`
        to work with a specific Python version or virtualenv.
    :param Project project: Provide a :class:`.Project` to make sure finding
        references works well, because the right folder is searched. There are
        also ways to modify the sys path and other things.
    """
    def __init__(self, code=None, line=None, column=None, path=None,
                 encoding=None, sys_path=None, environment=None,
                 project=None, source=None):
        self._orig_path = path
        # An empty path (also empty string) should always result in no path.
        self.path = os.path.abspath(path) if path else None

        if encoding is None:
            encoding = 'utf-8'
        else:
            warnings.warn(
                "Deprecated since version 0.17.0. You should cast to valid "
                "unicode yourself, especially if you are not using utf-8.",
                DeprecationWarning,
                stacklevel=2
            )
        if line is not None:
            warnings.warn(
                "Providing the line is now done in the functions themselves "
                "like `Script(...).complete(line, column)`",
                DeprecationWarning,
                stacklevel=2
            )
        if column is not None:
            warnings.warn(
                "Providing the column is now done in the functions themselves "
                "like `Script(...).complete(line, column)`",
                DeprecationWarning,
                stacklevel=2
            )
        if source is not None:
            code = source
            warnings.warn(
                "Use the code keyword argument instead.",
                DeprecationWarning,
                stacklevel=2
            )
        if code is None:
            # TODO add a better warning than the traceback!
            with open(path, 'rb') as f:
                code = f.read()

        if sys_path is not None and not is_py3:
            sys_path = list(map(force_unicode, sys_path))

        if project is None:
            # Load the Python grammar of the current interpreter.
            project = get_default_project(
                os.path.dirname(self.path) if path else None
            )
        # TODO deprecate and remove sys_path from the Script API.
        if sys_path is not None:
            project._sys_path = sys_path
            warnings.warn(
                "Deprecated since version 0.17.0. Use the project API instead, "
                "which means Script(project=Project(dir, sys_path=sys_path)) instead.",
                DeprecationWarning,
                stacklevel=2
            )

        self._inference_state = InferenceState(
            project, environment=environment, script_path=self.path
        )
        debug.speed('init')
        self._module_node, code = self._inference_state.parse_and_get_code(
            code=code,
            path=self.path,
            encoding=encoding,
            use_latest_grammar=path and path.endswith('.pyi'),
            cache=False,  # No disk cache, because the current script often changes.
            diff_cache=settings.fast_parser,
            cache_path=settings.cache_directory,
        )
        debug.speed('parsed')
        self._code_lines = parso.split_lines(code, keepends=True)
        self._code = code
        self._pos = line, column

        cache.clear_time_caches()
        debug.reset_time()

    # Cache the module, this is mostly useful for testing, since this shouldn't
    # be called multiple times.
    @cache.memoize_method
    def _get_module(self):
        names = None
        is_package = False
        if self.path is not None:
            import_names, is_p = transform_path_to_dotted(
                self._inference_state.get_sys_path(add_parent_paths=False),
                self.path
            )
            if import_names is not None:
                names = import_names
                is_package = is_p

        if self.path is None:
            file_io = None
        else:
            file_io = KnownContentFileIO(cast_path(self.path), self._code)
        if self.path is not None and self.path.endswith('.pyi'):
            # We are in a stub file. Try to load the stub properly.
            stub_module = load_proper_stub_module(
                self._inference_state,
                file_io,
                names,
                self._module_node
            )
            if stub_module is not None:
                return stub_module

        if names is None:
            names = ('__main__',)

        module = ModuleValue(
            self._inference_state, self._module_node,
            file_io=file_io,
            string_names=names,
            code_lines=self._code_lines,
            is_package=is_package,
        )
        if names[0] not in ('builtins', '__builtin__', 'typing'):
            # These modules are essential for Jedi, so don't overwrite them.
            self._inference_state.module_cache.add(names, ValueSet([module]))
        return module

    def _get_module_context(self):
        return self._get_module().as_context()

    def __repr__(self):
        return '<%s: %s %r>' % (
            self.__class__.__name__,
            repr(self._orig_path),
            self._inference_state.environment,
        )

    @validate_line_column
    def complete(self, line=None, column=None, **kwargs):
        """
        Completes objects under the cursor.

        Those objects contain information about the completions, more than just
        names.

        :param fuzzy: Default False. Will return fuzzy completions, which means
            that e.g. ``ooa`` will match ``foobar``.
        :return: Completion objects, sorted by name. Normal names appear
            before "private" names that start with ``_`` and those appear
            before magic methods and name mangled names that start with ``__``.
        :rtype: list of :class:`.Completion`
        """
        return self._complete(line, column, **kwargs)

    def _complete(self, line, column, fuzzy=False):  # Python 2...
        with debug.increase_indent_cm('complete'):
            completion = Completion(
                self._inference_state, self._get_module_context(), self._code_lines,
                (line, column), self.get_signatures, fuzzy=fuzzy,
            )
            return completion.complete()

    def completions(self, fuzzy=False):
        warnings.warn(
            "Deprecated since version 0.16.0. Use Script(...).complete instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.complete(*self._pos, fuzzy=fuzzy)

    @validate_line_column
    def infer(self, line=None, column=None, **kwargs):
        """
        Return the definitions of under the cursor. It is basically a wrapper
        around Jedi's type inference.

        This method follows complicated paths and returns the end, not the
        first definition. The big difference between :meth:`goto` and
        :meth:`infer` is that :meth:`goto` doesn't
        follow imports and statements. Multiple objects may be returned,
        because depending on an option you can have two different versions of a
        function.

        :param only_stubs: Only return stubs for this method.
        :param prefer_stubs: Prefer stubs to Python objects for this method.
        :rtype: list of :class:`.Name`
        """
        with debug.increase_indent_cm('infer'):
            return self._infer(line, column, **kwargs)

    def goto_definitions(self, **kwargs):
        warnings.warn(
            "Deprecated since version 0.16.0. Use Script(...).infer instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.infer(*self._pos, **kwargs)

    def _infer(self, line, column, only_stubs=False, prefer_stubs=False):
        pos = line, column
        leaf = self._module_node.get_name_of_position(pos)
        if leaf is None:
            leaf = self._module_node.get_leaf_for_position(pos)
            if leaf is None or leaf.type == 'string':
                return []

        context = self._get_module_context().create_context(leaf)

        values = helpers.infer(self._inference_state, context, leaf)
        values = convert_values(
            values,
            only_stubs=only_stubs,
            prefer_stubs=prefer_stubs,
        )

        defs = [classes.Name(self._inference_state, c.name) for c in values]
        # The additional set here allows the definitions to become unique in an
        # API sense. In the internals we want to separate more things than in
        # the API.
        return helpers.sorted_definitions(set(defs))

    def goto_assignments(self, follow_imports=False, follow_builtin_imports=False, **kwargs):
        warnings.warn(
            "Deprecated since version 0.16.0. Use Script(...).goto instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.goto(*self._pos,
                         follow_imports=follow_imports,
                         follow_builtin_imports=follow_builtin_imports,
                         **kwargs)

    @validate_line_column
    def goto(self, line=None, column=None, **kwargs):
        """
        Goes to the name that defined the object under the cursor. Optionally
        you can follow imports.
        Multiple objects may be returned, depending on an if you can have two
        different versions of a function.

        :param follow_imports: The method will follow imports.
        :param follow_builtin_imports: If ``follow_imports`` is True will try
            to look up names in builtins (i.e. compiled or extension modules).
        :param only_stubs: Only return stubs for this method.
        :param prefer_stubs: Prefer stubs to Python objects for this method.
        :rtype: list of :class:`.Name`
        """
        with debug.increase_indent_cm('goto'):
            return self._goto(line, column, **kwargs)

    def _goto(self, line, column, follow_imports=False, follow_builtin_imports=False,
              only_stubs=False, prefer_stubs=False):
        tree_name = self._module_node.get_name_of_position((line, column))
        if tree_name is None:
            # Without a name we really just want to jump to the result e.g.
            # executed by `foo()`, if we the cursor is after `)`.
            return self.infer(line, column, only_stubs=only_stubs, prefer_stubs=prefer_stubs)
        name = self._get_module_context().create_name(tree_name)

        # Make it possible to goto the super class function/attribute
        # definitions, when they are overwritten.
        names = []
        if name.tree_name.is_definition() and name.parent_context.is_class():
            class_node = name.parent_context.tree_node
            class_value = self._get_module_context().create_value(class_node)
            mro = class_value.py__mro__()
            next(mro)  # Ignore the first entry, because it's the class itself.
            for cls in mro:
                names = cls.goto(tree_name.value)
                if names:
                    break

        if not names:
            names = list(name.goto())

        if follow_imports:
            names = helpers.filter_follow_imports(names, follow_builtin_imports)
        names = convert_names(
            names,
            only_stubs=only_stubs,
            prefer_stubs=prefer_stubs,
        )

        defs = [classes.Name(self._inference_state, d) for d in set(names)]
        # Avoid duplicates
        return list(set(helpers.sorted_definitions(defs)))

    @_no_python2_support
    def search(self, string, **kwargs):
        """
        Searches a name in the current file. For a description of how the
        search string should look like, please have a look at
        :meth:`.Project.search`.

        :param bool all_scopes: Default False; searches not only for
            definitions on the top level of a module level, but also in
            functions and classes.
        :yields: :class:`.Name`
        """
        return self._search(string, **kwargs)  # Python 2 ...

    def _search(self, string, all_scopes=False):
        return self._search_func(string, all_scopes=all_scopes)

    @to_list
    def _search_func(self, string, all_scopes=False, complete=False, fuzzy=False):
        names = self._names(all_scopes=all_scopes)
        wanted_type, wanted_names = helpers.split_search_string(string)
        return search_in_module(
            self._inference_state,
            self._get_module_context(),
            names=names,
            wanted_type=wanted_type,
            wanted_names=wanted_names,
            complete=complete,
            fuzzy=fuzzy,
        )

    def complete_search(self, string, **kwargs):
        """
        Like :meth:`.Script.search`, but completes that string. If you want to
        have all possible definitions in a file you can also provide an empty
        string.

        :param bool all_scopes: Default False; searches not only for
            definitions on the top level of a module level, but also in
            functions and classes.
        :param fuzzy: Default False. Will return fuzzy completions, which means
            that e.g. ``ooa`` will match ``foobar``.
        :yields: :class:`.Completion`
        """
        return self._search_func(string, complete=True, **kwargs)

    @validate_line_column
    def help(self, line=None, column=None):
        """
        Used to display a help window to users.  Uses :meth:`.Script.goto` and
        returns additional definitions for keywords and operators.

        Typically you will want to display :meth:`.BaseName.docstring` to the
        user for all the returned definitions.

        The additional definitions are ``Name(...).type == 'keyword'``.
        These definitions do not have a lot of value apart from their docstring
        attribute, which contains the output of Python's :func:`help` function.

        :rtype: list of :class:`.Name`
        """
        definitions = self.goto(line, column, follow_imports=True)
        if definitions:
            return definitions
        leaf = self._module_node.get_leaf_for_position((line, column))
        if leaf is not None and leaf.type in ('keyword', 'operator', 'error_leaf'):
            def need_pydoc():
                if leaf.value in ('(', ')', '[', ']'):
                    if leaf.parent.type == 'trailer':
                        return False
                    if leaf.parent.type == 'atom':
                        return False
                grammar = self._inference_state.grammar
                # This parso stuff is not public, but since I control it, this
                # is fine :-) ~dave
                reserved = grammar._pgen_grammar.reserved_syntax_strings.keys()
                return leaf.value in reserved

            if need_pydoc():
                name = KeywordName(self._inference_state, leaf.value)
                return [classes.Name(self._inference_state, name)]
        return []

    def usages(self, **kwargs):
        warnings.warn(
            "Deprecated since version 0.16.0. Use Script(...).get_references instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.get_references(*self._pos, **kwargs)

    @validate_line_column
    def get_references(self, line=None, column=None, **kwargs):
        """
        Lists all references of a variable in a project. Since this can be
        quite hard to do for Jedi, if it is too complicated, Jedi will stop
        searching.

        :param include_builtins: Default ``True``. If ``False``, checks if a reference
            is a builtin (e.g. ``sys``) and in that case does not return it.
        :param scope: Default ``'project'``. If ``'file'``, include references in
            the current module only.
        :rtype: list of :class:`.Name`
        """

        def _references(include_builtins=True, scope='project'):
            if scope not in ('project', 'file'):
                raise ValueError('Only the scopes "file" and "project" are allowed')
            tree_name = self._module_node.get_name_of_position((line, column))
            if tree_name is None:
                # Must be syntax
                return []

            names = find_references(self._get_module_context(), tree_name, scope == 'file')

            definitions = [classes.Name(self._inference_state, n) for n in names]
            if not include_builtins or scope == 'file':
                definitions = [d for d in definitions if not d.in_builtin_module()]
            return helpers.sorted_definitions(definitions)
        return _references(**kwargs)

    def call_signatures(self):
        warnings.warn(
            "Deprecated since version 0.16.0. Use Script(...).get_signatures instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.get_signatures(*self._pos)

    @validate_line_column
    def get_signatures(self, line=None, column=None):
        """
        Return the function object of the call under the cursor.

        E.g. if the cursor is here::

            abs(# <-- cursor is here

        This would return the ``abs`` function. On the other hand::

            abs()# <-- cursor is here

        This would return an empty list..

        :rtype: list of :class:`.Signature`
        """
        pos = line, column
        call_details = helpers.get_signature_details(self._module_node, pos)
        if call_details is None:
            return []

        context = self._get_module_context().create_context(call_details.bracket_leaf)
        definitions = helpers.cache_signatures(
            self._inference_state,
            context,
            call_details.bracket_leaf,
            self._code_lines,
            pos
        )
        debug.speed('func_call followed')

        # TODO here we use stubs instead of the actual values. We should use
        # the signatures from stubs, but the actual values, probably?!
        return [classes.Signature(self._inference_state, signature, call_details)
                for signature in definitions.get_signatures()]

    @validate_line_column
    def get_context(self, line=None, column=None):
        """
        Returns the scope context under the cursor. This basically means the
        function, class or module where the cursor is at.

        :rtype: :class:`.Name`
        """
        pos = (line, column)
        leaf = self._module_node.get_leaf_for_position(pos, include_prefixes=True)
        if leaf.start_pos > pos or leaf.type == 'endmarker':
            previous_leaf = leaf.get_previous_leaf()
            if previous_leaf is not None:
                leaf = previous_leaf

        module_context = self._get_module_context()

        n = tree.search_ancestor(leaf, 'funcdef', 'classdef')
        if n is not None and n.start_pos < pos <= n.children[-1].start_pos:
            # This is a bit of a special case. The context of a function/class
            # name/param/keyword is always it's parent context, not the
            # function itself. Catch all the cases here where we are before the
            # suite object, but still in the function.
            context = module_context.create_value(n).as_context()
        else:
            context = module_context.create_context(leaf)

        while context.name is None:
            context = context.parent_context  # comprehensions

        definition = classes.Name(self._inference_state, context.name)
        while definition.type != 'module':
            name = definition._name  # TODO private access
            tree_name = name.tree_name
            if tree_name is not None:  # Happens with lambdas.
                scope = tree_name.get_definition()
                if scope.start_pos[1] < column:
                    break
            definition = definition.parent()
        return definition

    def _analysis(self):
        self._inference_state.is_analysis = True
        self._inference_state.analysis_modules = [self._module_node]
        module = self._get_module_context()
        try:
            for node in get_executable_nodes(self._module_node):
                context = module.create_context(node)
                if node.type in ('funcdef', 'classdef'):
                    # Resolve the decorators.
                    tree_name_to_values(self._inference_state, context, node.children[1])
                elif isinstance(node, tree.Import):
                    import_names = set(node.get_defined_names())
                    if node.is_nested():
                        import_names |= set(path[-1] for path in node.get_paths())
                    for n in import_names:
                        imports.infer_import(context, n)
                elif node.type == 'expr_stmt':
                    types = context.infer_node(node)
                    for testlist in node.children[:-1:2]:
                        # Iterate tuples.
                        unpack_tuple_to_dict(context, types, testlist)
                else:
                    if node.type == 'name':
                        defs = self._inference_state.infer(context, node)
                    else:
                        defs = infer_call_of_leaf(context, node)
                    try_iter_content(defs)
                self._inference_state.reset_recursion_limitations()

            ana = [a for a in self._inference_state.analysis if self.path == a.path]
            return sorted(set(ana), key=lambda x: x.line)
        finally:
            self._inference_state.is_analysis = False

    def get_names(self, **kwargs):
        """
        Returns names defined in the current file.

        :param all_scopes: If True lists the names of all scopes instead of
            only the module namespace.
        :param definitions: If True lists the names that have been defined by a
            class, function or a statement (``a = b`` returns ``a``).
        :param references: If True lists all the names that are not listed by
            ``definitions=True``. E.g. ``a = b`` returns ``b``.
        :rtype: list of :class:`.Name`
        """
        names = self._names(**kwargs)
        return [classes.Name(self._inference_state, n) for n in names]

    def get_syntax_errors(self):
        """
        Lists all syntax errors in the current file.

        :rtype: list of :class:`.SyntaxError`
        """
        return parso_to_jedi_errors(self._inference_state.grammar, self._module_node)

    def _names(self, all_scopes=False, definitions=True, references=False):
        # Set line/column to a random position, because they don't matter.
        module_context = self._get_module_context()
        defs = [
            module_context.create_name(name)
            for name in helpers.get_module_names(
                self._module_node,
                all_scopes=all_scopes,
                definitions=definitions,
                references=references,
            )
        ]
        return sorted(defs, key=lambda x: x.start_pos)

    @_no_python2_support
    def rename(self, line=None, column=None, **kwargs):
        """
        Renames all references of the variable under the cursor.

        :param new_name: The variable under the cursor will be renamed to this
            string.
        :raises: :exc:`.RefactoringError`
        :rtype: :class:`.Refactoring`
        """
        return self._rename(line, column, **kwargs)

    def _rename(self, line, column, new_name):  # Python 2...
        definitions = self.get_references(line, column, include_builtins=False)
        return refactoring.rename(self._inference_state, definitions, new_name)

    @_no_python2_support
    def extract_variable(self, line, column, **kwargs):
        """
        Moves an expression to a new statemenet.

        For example if you have the cursor on ``foo`` and provide a
        ``new_name`` called ``bar``::

            foo = 3.1
            x = int(foo + 1)

        the code above will become::

            foo = 3.1
            bar = foo + 1
            x = int(bar)

        :param new_name: The expression under the cursor will be renamed to
            this string.
        :param int until_line: The the selection range ends at this line, when
            omitted, Jedi will be clever and try to define the range itself.
        :param int until_column: The the selection range ends at this column, when
            omitted, Jedi will be clever and try to define the range itself.
        :raises: :exc:`.RefactoringError`
        :rtype: :class:`.Refactoring`
        """
        return self._extract_variable(line, column, **kwargs)  # Python 2...

    @validate_line_column
    def _extract_variable(self, line, column, new_name, until_line=None, until_column=None):
        if until_line is None and until_column is None:
            until_pos = None
        else:
            if until_line is None:
                until_line = line
            if until_column is None:
                until_column = len(self._code_lines[until_line - 1])
            until_pos = until_line, until_column
        return extract_variable(
            self._inference_state, self.path, self._module_node,
            new_name, (line, column), until_pos
        )

    @_no_python2_support
    def extract_function(self, line, column, **kwargs):
        """
        Moves an expression to a new function.

        For example if you have the cursor on ``foo`` and provide a
        ``new_name`` called ``bar``::

            global_var = 3

            def x():
                foo = 3.1
                x = int(foo + 1 + global_var)

        the code above will become::

            global_var = 3

            def bar(foo):
                return int(foo + 1 + global_var)

            def x():
                foo = 3.1
                x = bar(foo)

        :param new_name: The expression under the cursor will be replaced with
            a function with this name.
        :param int until_line: The the selection range ends at this line, when
            omitted, Jedi will be clever and try to define the range itself.
        :param int until_column: The the selection range ends at this column, when
            omitted, Jedi will be clever and try to define the range itself.
        :raises: :exc:`.RefactoringError`
        :rtype: :class:`.Refactoring`
        """
        return self._extract_function(line, column, **kwargs)  # Python 2...

    @validate_line_column
    def _extract_function(self, line, column, new_name, until_line=None, until_column=None):
        if until_line is None and until_column is None:
            until_pos = None
        else:
            if until_line is None:
                until_line = line
            if until_column is None:
                until_column = len(self._code_lines[until_line - 1])
            until_pos = until_line, until_column
        return extract_function(
            self._inference_state, self.path, self._get_module_context(),
            new_name, (line, column), until_pos
        )

    @_no_python2_support
    def inline(self, line=None, column=None):
        """
        Inlines a variable under the cursor. This is basically the opposite of
        extracting a variable. For example with the cursor on bar::

            foo = 3.1
            bar = foo + 1
            x = int(bar)

        the code above will become::

            foo = 3.1
            x = int(foo + 1)

        :raises: :exc:`.RefactoringError`
        :rtype: :class:`.Refactoring`
        """
        names = [d._name for d in self.get_references(line, column, include_builtins=True)]
        return refactoring.inline(self._inference_state, names)


class Interpreter(Script):
    """
    Jedi's API for Python REPLs.

    Implements all of the methods that are present in :class:`.Script` as well.

    In addition to completions that normal REPL completion does like
    ``str.upper``, Jedi also supports code completion based on static code
    analysis. For example Jedi will complete ``str().upper``.

    >>> from os.path import join
    >>> namespace = locals()
    >>> script = Interpreter('join("").up', [namespace])
    >>> print(script.complete()[0].name)
    upper

    All keyword arguments are same as the arguments for :class:`.Script`.

    :param str code: Code to parse.
    :type namespaces: typing.List[dict]
    :param namespaces: A list of namespace dictionaries such as the one
        returned by :func:`globals` and :func:`locals`.
    """
    _allow_descriptor_getattr_default = True

    def __init__(self, code, namespaces, **kwds):
        try:
            namespaces = [dict(n) for n in namespaces]
        except Exception:
            raise TypeError("namespaces must be a non-empty list of dicts.")

        environment = kwds.get('environment', None)
        if environment is None:
            environment = InterpreterEnvironment()
        else:
            if not isinstance(environment, InterpreterEnvironment):
                raise TypeError("The environment needs to be an InterpreterEnvironment subclass.")

        super(Interpreter, self).__init__(code, environment=environment,
                                          project=Project(os.getcwd()), **kwds)
        self.namespaces = namespaces
        self._inference_state.allow_descriptor_getattr = self._allow_descriptor_getattr_default

    @cache.memoize_method
    def _get_module_context(self):
        tree_module_value = ModuleValue(
            self._inference_state, self._module_node,
            file_io=KnownContentFileIO(self.path, self._code),
            string_names=('__main__',),
            code_lines=self._code_lines,
        )
        return interpreter.MixedModuleContext(
            tree_module_value,
            self.namespaces,
        )


def names(source=None, path=None, encoding='utf-8', all_scopes=False,
          definitions=True, references=False, environment=None):
    warnings.warn(
        "Deprecated since version 0.16.0. Use Script(...).get_names instead.",
        DeprecationWarning,
        stacklevel=2
    )

    return Script(source, path=path, encoding=encoding).get_names(
        all_scopes=all_scopes,
        definitions=definitions,
        references=references,
    )


def preload_module(*modules):
    """
    Preloading modules tells Jedi to load a module now, instead of lazy parsing
    of modules. This can be useful for IDEs, to control which modules to load
    on startup.

    :param modules: different module names, list of string.
    """
    for m in modules:
        s = "import %s as x; x." % m
        Script(s, path=None).complete(1, len(s))


def set_debug_function(func_cb=debug.print_to_stdout, warnings=True,
                       notices=True, speed=True):
    """
    Define a callback debug function to get all the debug messages.

    If you don't specify any arguments, debug messages will be printed to stdout.

    :param func_cb: The callback function for debug messages.
    """
    debug.debug_function = func_cb
    debug.enable_warning = warnings
    debug.enable_notice = notices
    debug.enable_speed = speed
