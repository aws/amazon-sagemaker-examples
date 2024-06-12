from __future__ import print_function
import sys
import os
import re
import inspect

from jedi._compatibility import find_module, cast_path, force_unicode, \
    all_suffixes, scandir
from jedi.inference.compiled import access
from jedi import debug
from jedi import parser_utils


def get_sys_path():
    return list(map(cast_path, sys.path))


def load_module(inference_state, **kwargs):
    return access.load_module(inference_state, **kwargs)


def get_compiled_method_return(inference_state, id, attribute, *args, **kwargs):
    handle = inference_state.compiled_subprocess.get_access_handle(id)
    return getattr(handle.access, attribute)(*args, **kwargs)


def create_simple_object(inference_state, obj):
    return access.create_access_path(inference_state, obj)


def get_module_info(inference_state, sys_path=None, full_name=None, **kwargs):
    """
    Returns Tuple[Union[NamespaceInfo, FileIO, None], Optional[bool]]
    """
    if sys_path is not None:
        sys.path, temp = sys_path, sys.path
    try:
        return find_module(full_name=full_name, **kwargs)
    except ImportError:
        return None, None
    finally:
        if sys_path is not None:
            sys.path = temp


def get_builtin_module_names(inference_state):
    return list(map(force_unicode, sys.builtin_module_names))


def _test_raise_error(inference_state, exception_type):
    """
    Raise an error to simulate certain problems for unit tests.
    """
    raise exception_type


def _test_print(inference_state, stderr=None, stdout=None):
    """
    Force some prints in the subprocesses. This exists for unit tests.
    """
    if stderr is not None:
        print(stderr, file=sys.stderr)
        sys.stderr.flush()
    if stdout is not None:
        print(stdout)
        sys.stdout.flush()


def _get_init_path(directory_path):
    """
    The __init__ file can be searched in a directory. If found return it, else
    None.
    """
    for suffix in all_suffixes():
        path = os.path.join(directory_path, '__init__' + suffix)
        if os.path.exists(path):
            return path
    return None


def safe_literal_eval(inference_state, value):
    return parser_utils.safe_literal_eval(value)


def iter_module_names(*args, **kwargs):
    return list(_iter_module_names(*args, **kwargs))


def _iter_module_names(inference_state, paths):
    # Python modules/packages
    for path in paths:
        try:
            dirs = scandir(path)
        except OSError:
            # The file might not exist or reading it might lead to an error.
            debug.warning("Not possible to list directory: %s", path)
            continue
        for dir_entry in dirs:
            name = dir_entry.name
            # First Namespaces then modules/stubs
            if dir_entry.is_dir():
                # pycache is obviously not an interestin namespace. Also the
                # name must be a valid identifier.
                # TODO use str.isidentifier, once Python 2 is removed
                if name != '__pycache__' and not re.search(r'\W|^\d', name):
                    yield name
            else:
                if name.endswith('.pyi'):  # Stub files
                    modname = name[:-4]
                else:
                    modname = inspect.getmodulename(name)

                if modname and '.' not in modname:
                    if modname != '__init__':
                        yield modname
