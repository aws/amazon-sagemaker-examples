"""schema is a library for validating Python data structures, such as those
obtained from config-files, forms, external services or command-line
parsing, converted from JSON/YAML (or something else) to Python data-types."""

import inspect
import re

try:
    from contextlib import ExitStack
except ImportError:
    from contextlib2 import ExitStack


__version__ = "0.7.5"
__all__ = [
    "Schema",
    "And",
    "Or",
    "Regex",
    "Optional",
    "Use",
    "Forbidden",
    "Const",
    "Literal",
    "SchemaError",
    "SchemaWrongKeyError",
    "SchemaMissingKeyError",
    "SchemaForbiddenKeyError",
    "SchemaUnexpectedTypeError",
    "SchemaOnlyOneAllowedError",
]


class SchemaError(Exception):
    """Error during Schema validation."""

    def __init__(self, autos, errors=None):
        self.autos = autos if type(autos) is list else [autos]
        self.errors = errors if type(errors) is list else [errors]
        Exception.__init__(self, self.code)

    @property
    def code(self):
        """
        Removes duplicates values in auto and error list.
        parameters.
        """

        def uniq(seq):
            """
            Utility function that removes duplicate.
            """
            seen = set()
            seen_add = seen.add
            # This way removes duplicates while preserving the order.
            return [x for x in seq if x not in seen and not seen_add(x)]

        data_set = uniq(i for i in self.autos if i is not None)
        error_list = uniq(i for i in self.errors if i is not None)
        if error_list:
            return "\n".join(error_list)
        return "\n".join(data_set)


class SchemaWrongKeyError(SchemaError):
    """Error Should be raised when an unexpected key is detected within the
    data set being."""

    pass


class SchemaMissingKeyError(SchemaError):
    """Error should be raised when a mandatory key is not found within the
    data set being validated"""

    pass


class SchemaOnlyOneAllowedError(SchemaError):
    """Error should be raised when an only_one Or key has multiple matching candidates"""

    pass


class SchemaForbiddenKeyError(SchemaError):
    """Error should be raised when a forbidden key is found within the
    data set being validated, and its value matches the value that was specified"""

    pass


class SchemaUnexpectedTypeError(SchemaError):
    """Error should be raised when a type mismatch is detected within the
    data set being validated."""

    pass


class And(object):
    """
    Utility function to combine validation directives in AND Boolean fashion.
    """

    def __init__(self, *args, **kw):
        self._args = args
        if not set(kw).issubset({"error", "schema", "ignore_extra_keys"}):
            diff = {"error", "schema", "ignore_extra_keys"}.difference(kw)
            raise TypeError("Unknown keyword arguments %r" % list(diff))
        self._error = kw.get("error")
        self._ignore_extra_keys = kw.get("ignore_extra_keys", False)
        # You can pass your inherited Schema class.
        self._schema = kw.get("schema", Schema)

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, ", ".join(repr(a) for a in self._args))

    @property
    def args(self):
        """The provided parameters"""
        return self._args

    def validate(self, data, **kwargs):
        """
        Validate data using defined sub schema/expressions ensuring all
        values are valid.
        :param data: to be validated with sub defined schemas.
        :return: returns validated data
        """
        for s in [self._schema(s, error=self._error, ignore_extra_keys=self._ignore_extra_keys) for s in self._args]:
            data = s.validate(data, **kwargs)
        return data


class Or(And):
    """Utility function to combine validation directives in a OR Boolean
    fashion."""

    def __init__(self, *args, **kwargs):
        self.only_one = kwargs.pop("only_one", False)
        self.match_count = 0
        super(Or, self).__init__(*args, **kwargs)

    def reset(self):
        failed = self.match_count > 1 and self.only_one
        self.match_count = 0
        if failed:
            raise SchemaOnlyOneAllowedError(["There are multiple keys present " + "from the %r condition" % self])

    def validate(self, data, **kwargs):
        """
        Validate data using sub defined schema/expressions ensuring at least
        one value is valid.
        :param data: data to be validated by provided schema.
        :return: return validated data if not validation
        """
        autos, errors = [], []
        for s in [self._schema(s, error=self._error, ignore_extra_keys=self._ignore_extra_keys) for s in self._args]:
            try:
                validation = s.validate(data, **kwargs)
                self.match_count += 1
                if self.match_count > 1 and self.only_one:
                    break
                return validation
            except SchemaError as _x:
                autos += _x.autos
                errors += _x.errors
        raise SchemaError(
            ["%r did not validate %r" % (self, data)] + autos,
            [self._error.format(data) if self._error else None] + errors,
        )


class Regex(object):
    """
    Enables schema.py to validate string using regular expressions.
    """

    # Map all flags bits to a more readable description
    NAMES = [
        "re.ASCII",
        "re.DEBUG",
        "re.VERBOSE",
        "re.UNICODE",
        "re.DOTALL",
        "re.MULTILINE",
        "re.LOCALE",
        "re.IGNORECASE",
        "re.TEMPLATE",
    ]

    def __init__(self, pattern_str, flags=0, error=None):
        self._pattern_str = pattern_str
        flags_list = [
            Regex.NAMES[i] for i, f in enumerate("{0:09b}".format(int(flags))) if f != "0"
        ]  # Name for each bit

        if flags_list:
            self._flags_names = ", flags=" + "|".join(flags_list)
        else:
            self._flags_names = ""

        self._pattern = re.compile(pattern_str, flags=flags)
        self._error = error

    def __repr__(self):
        return "%s(%r%s)" % (self.__class__.__name__, self._pattern_str, self._flags_names)

    @property
    def pattern_str(self):
        """The pattern for the represented regular expression"""
        return self._pattern_str

    def validate(self, data, **kwargs):
        """
        Validated data using defined regex.
        :param data: data to be validated
        :return: return validated data.
        """
        e = self._error

        try:
            if self._pattern.search(data):
                return data
            else:
                raise SchemaError("%r does not match %r" % (self, data), e.format(data) if e else None)
        except TypeError:
            raise SchemaError("%r is not string nor buffer" % data, e)


class Use(object):
    """
    For more general use cases, you can use the Use class to transform
    the data while it is being validate.
    """

    def __init__(self, callable_, error=None):
        if not callable(callable_):
            raise TypeError("Expected a callable, not %r" % callable_)
        self._callable = callable_
        self._error = error

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self._callable)

    def validate(self, data, **kwargs):
        try:
            return self._callable(data)
        except SchemaError as x:
            raise SchemaError([None] + x.autos, [self._error.format(data) if self._error else None] + x.errors)
        except BaseException as x:
            f = _callable_str(self._callable)
            raise SchemaError("%s(%r) raised %r" % (f, data, x), self._error.format(data) if self._error else None)


COMPARABLE, CALLABLE, VALIDATOR, TYPE, DICT, ITERABLE = range(6)


def _priority(s):
    """Return priority for a given object."""
    if type(s) in (list, tuple, set, frozenset):
        return ITERABLE
    if type(s) is dict:
        return DICT
    if issubclass(type(s), type):
        return TYPE
    if isinstance(s, Literal):
        return COMPARABLE
    if hasattr(s, "validate"):
        return VALIDATOR
    if callable(s):
        return CALLABLE
    else:
        return COMPARABLE


def _invoke_with_optional_kwargs(f, **kwargs):
    s = inspect.signature(f)
    if len(s.parameters) == 0:
        return f()
    return f(**kwargs)


class Schema(object):
    """
    Entry point of the library, use this class to instantiate validation
    schema for the data that will be validated.
    """

    def __init__(self, schema, error=None, ignore_extra_keys=False, name=None, description=None, as_reference=False):
        self._schema = schema
        self._error = error
        self._ignore_extra_keys = ignore_extra_keys
        self._name = name
        self._description = description
        # Ask json_schema to create a definition for this schema and use it as part of another
        self.as_reference = as_reference
        if as_reference and name is None:
            raise ValueError("Schema used as reference should have a name")

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self._schema)

    @property
    def schema(self):
        return self._schema

    @property
    def description(self):
        return self._description

    @property
    def name(self):
        return self._name

    @property
    def ignore_extra_keys(self):
        return self._ignore_extra_keys

    @staticmethod
    def _dict_key_priority(s):
        """Return priority for a given key object."""
        if isinstance(s, Hook):
            return _priority(s._schema) - 0.5
        if isinstance(s, Optional):
            return _priority(s._schema) + 0.5
        return _priority(s)

    @staticmethod
    def _is_optional_type(s):
        """Return True if the given key is optional (does not have to be found)"""
        return any(isinstance(s, optional_type) for optional_type in [Optional, Hook])

    def is_valid(self, data, **kwargs):
        """Return whether the given data has passed all the validations
        that were specified in the given schema.
        """
        try:
            self.validate(data, **kwargs)
        except SchemaError:
            return False
        else:
            return True

    def _prepend_schema_name(self, message):
        """
        If a custom schema name has been defined, prepends it to the error
        message that gets raised when a schema error occurs.
        """
        if self._name:
            message = "{0!r} {1!s}".format(self._name, message)
        return message

    def validate(self, data, **kwargs):
        Schema = self.__class__
        s = self._schema
        e = self._error
        i = self._ignore_extra_keys

        if isinstance(s, Literal):
            s = s.schema

        flavor = _priority(s)
        if flavor == ITERABLE:
            data = Schema(type(s), error=e).validate(data, **kwargs)
            o = Or(*s, error=e, schema=Schema, ignore_extra_keys=i)
            return type(data)(o.validate(d, **kwargs) for d in data)
        if flavor == DICT:
            exitstack = ExitStack()
            data = Schema(dict, error=e).validate(data, **kwargs)
            new = type(data)()  # new - is a dict of the validated values
            coverage = set()  # matched schema keys
            # for each key and value find a schema entry matching them, if any
            sorted_skeys = sorted(s, key=self._dict_key_priority)
            for skey in sorted_skeys:
                if hasattr(skey, "reset"):
                    exitstack.callback(skey.reset)

            with exitstack:
                # Evaluate dictionaries last
                data_items = sorted(data.items(), key=lambda value: isinstance(value[1], dict))
                for key, value in data_items:
                    for skey in sorted_skeys:
                        svalue = s[skey]
                        try:
                            nkey = Schema(skey, error=e).validate(key, **kwargs)
                        except SchemaError:
                            pass
                        else:
                            if isinstance(skey, Hook):
                                # As the content of the value makes little sense for
                                # keys with a hook, we reverse its meaning:
                                # we will only call the handler if the value does match
                                # In the case of the forbidden key hook,
                                # we will raise the SchemaErrorForbiddenKey exception
                                # on match, allowing for excluding a key only if its
                                # value has a certain type, and allowing Forbidden to
                                # work well in combination with Optional.
                                try:
                                    nvalue = Schema(svalue, error=e).validate(value, **kwargs)
                                except SchemaError:
                                    continue
                                skey.handler(nkey, data, e)
                            else:
                                try:
                                    nvalue = Schema(svalue, error=e, ignore_extra_keys=i).validate(value, **kwargs)
                                except SchemaError as x:
                                    k = "Key '%s' error:" % nkey
                                    message = self._prepend_schema_name(k)
                                    raise SchemaError([message] + x.autos, [e.format(data) if e else None] + x.errors)
                                else:
                                    new[nkey] = nvalue
                                    coverage.add(skey)
                                    break
            required = set(k for k in s if not self._is_optional_type(k))
            if not required.issubset(coverage):
                missing_keys = required - coverage
                s_missing_keys = ", ".join(repr(k) for k in sorted(missing_keys, key=repr))
                message = "Missing key%s: %s" % (_plural_s(missing_keys), s_missing_keys)
                message = self._prepend_schema_name(message)
                raise SchemaMissingKeyError(message, e.format(data) if e else None)
            if not self._ignore_extra_keys and (len(new) != len(data)):
                wrong_keys = set(data.keys()) - set(new.keys())
                s_wrong_keys = ", ".join(repr(k) for k in sorted(wrong_keys, key=repr))
                message = "Wrong key%s %s in %r" % (_plural_s(wrong_keys), s_wrong_keys, data)
                message = self._prepend_schema_name(message)
                raise SchemaWrongKeyError(message, e.format(data) if e else None)

            # Apply default-having optionals that haven't been used:
            defaults = set(k for k in s if isinstance(k, Optional) and hasattr(k, "default")) - coverage
            for default in defaults:
                new[default.key] = _invoke_with_optional_kwargs(default.default, **kwargs) if callable(default.default) else default.default

            return new
        if flavor == TYPE:
            if isinstance(data, s) and not (isinstance(data, bool) and s == int):
                return data
            else:
                message = "%r should be instance of %r" % (data, s.__name__)
                message = self._prepend_schema_name(message)
                raise SchemaUnexpectedTypeError(message, e.format(data) if e else None)
        if flavor == VALIDATOR:
            try:
                return s.validate(data, **kwargs)
            except SchemaError as x:
                raise SchemaError([None] + x.autos, [e.format(data) if e else None] + x.errors)
            except BaseException as x:
                message = "%r.validate(%r) raised %r" % (s, data, x)
                message = self._prepend_schema_name(message)
                raise SchemaError(message, e.format(data) if e else None)
        if flavor == CALLABLE:
            f = _callable_str(s)
            try:
                if s(data):
                    return data
            except SchemaError as x:
                raise SchemaError([None] + x.autos, [e.format(data) if e else None] + x.errors)
            except BaseException as x:
                message = "%s(%r) raised %r" % (f, data, x)
                message = self._prepend_schema_name(message)
                raise SchemaError(message, e.format(data) if e else None)
            message = "%s(%r) should evaluate to True" % (f, data)
            message = self._prepend_schema_name(message)
            raise SchemaError(message, e.format(data) if e else None)
        if s == data:
            return data
        else:
            message = "%r does not match %r" % (s, data)
            message = self._prepend_schema_name(message)
            raise SchemaError(message, e.format(data) if e else None)

    def json_schema(self, schema_id, use_refs=False, **kwargs):
        """Generate a draft-07 JSON schema dict representing the Schema.
        This method must be called with a schema_id.

        :param schema_id: The value of the $id on the main schema
        :param use_refs: Enable reusing object references in the resulting JSON schema.
                         Schemas with references are harder to read by humans, but are a lot smaller when there
                         is a lot of reuse
        """

        seen = dict()  # For use_refs
        definitions_by_name = {}

        def _json_schema(schema, is_main_schema=True, description=None, allow_reference=True):
            Schema = self.__class__

            def _create_or_use_ref(return_dict):
                """If not already seen, return the provided part of the schema unchanged.
                If already seen, give an id to the already seen dict and return a reference to the previous part
                of the schema instead.
                """
                if not use_refs or is_main_schema:
                    return return_schema

                hashed = hash(repr(sorted(return_dict.items())))

                if hashed not in seen:
                    seen[hashed] = return_dict
                    return return_dict
                else:
                    id_str = "#" + str(hashed)
                    seen[hashed]["$id"] = id_str
                    return {"$ref": id_str}

            def _get_type_name(python_type):
                """Return the JSON schema name for a Python type"""
                if python_type == str:
                    return "string"
                elif python_type == int:
                    return "integer"
                elif python_type == float:
                    return "number"
                elif python_type == bool:
                    return "boolean"
                elif python_type == list:
                    return "array"
                elif python_type == dict:
                    return "object"
                return "string"

            def _to_json_type(value):
                """Attempt to convert a constant value (for "const" and "default") to a JSON serializable value"""
                if value is None or type(value) in (str, int, float, bool, list, dict):
                    return value

                if type(value) in (tuple, set, frozenset):
                    return list(value)

                if isinstance(value, Literal):
                    return value.schema

                return str(value)

            def _to_schema(s, ignore_extra_keys):
                if not isinstance(s, Schema):
                    return Schema(s, ignore_extra_keys=ignore_extra_keys)

                return s

            s = schema.schema
            i = schema.ignore_extra_keys
            flavor = _priority(s)

            return_schema = {}

            return_description = description or schema.description
            if return_description:
                return_schema["description"] = return_description

            # Check if we have to create a common definition and use as reference
            if allow_reference and schema.as_reference:
                # Generate sub schema if not already done
                if schema.name not in definitions_by_name:
                    definitions_by_name[schema.name] = {}  # Avoid infinite loop
                    definitions_by_name[schema.name] = _json_schema(schema, is_main_schema=False, allow_reference=False)

                return_schema["$ref"] = "#/definitions/" + schema.name
            else:
                if flavor == TYPE:
                    # Handle type
                    return_schema["type"] = _get_type_name(s)
                elif flavor == ITERABLE:
                    # Handle arrays or dict schema

                    return_schema["type"] = "array"
                    if len(s) == 1:
                        return_schema["items"] = _json_schema(_to_schema(s[0], i), is_main_schema=False)
                    elif len(s) > 1:
                        return_schema["items"] = _json_schema(Schema(Or(*s)), is_main_schema=False)
                elif isinstance(s, Or):
                    # Handle Or values

                    # Check if we can use an enum
                    if all(priority == COMPARABLE for priority in [_priority(value) for value in s.args]):
                        or_values = [str(s) if isinstance(s, Literal) else s for s in s.args]
                        # All values are simple, can use enum or const
                        if len(or_values) == 1:
                            return_schema["const"] = _to_json_type(or_values[0])
                            return return_schema
                        return_schema["enum"] = or_values
                    else:
                        # No enum, let's go with recursive calls
                        any_of_values = []
                        for or_key in s.args:
                            new_value = _json_schema(_to_schema(or_key, i), is_main_schema=False)
                            if new_value != {} and new_value not in any_of_values:
                                any_of_values.append(new_value)
                        if len(any_of_values) == 1:
                            # Only one representable condition remains, do not put under anyOf
                            return_schema.update(any_of_values[0])
                        else:
                            return_schema["anyOf"] = any_of_values
                elif isinstance(s, And):
                    # Handle And values
                    all_of_values = []
                    for and_key in s.args:
                        new_value = _json_schema(_to_schema(and_key, i), is_main_schema=False)
                        if new_value != {} and new_value not in all_of_values:
                            all_of_values.append(new_value)
                    if len(all_of_values) == 1:
                        # Only one representable condition remains, do not put under allOf
                        return_schema.update(all_of_values[0])
                    else:
                        return_schema["allOf"] = all_of_values
                elif flavor == COMPARABLE:
                    return_schema["const"] = _to_json_type(s)
                elif flavor == VALIDATOR and type(s) == Regex:
                    return_schema["type"] = "string"
                    return_schema["pattern"] = s.pattern_str
                else:
                    if flavor != DICT:
                        # If not handled, do not check
                        return return_schema

                    # Schema is a dict

                    required_keys = []
                    expanded_schema = {}
                    additional_properties = i
                    for key in s:
                        if isinstance(key, Hook):
                            continue

                        def _key_allows_additional_properties(key):
                            """Check if a key is broad enough to allow additional properties"""
                            if isinstance(key, Optional):
                                return _key_allows_additional_properties(key.schema)

                            return key == str or key == object

                        def _get_key_description(key):
                            """Get the description associated to a key (as specified in a Literal object). Return None if not a Literal"""
                            if isinstance(key, Optional):
                                return _get_key_description(key.schema)

                            if isinstance(key, Literal):
                                return key.description

                            return None

                        def _get_key_name(key):
                            """Get the name of a key (as specified in a Literal object). Return the key unchanged if not a Literal"""
                            if isinstance(key, Optional):
                                return _get_key_name(key.schema)

                            if isinstance(key, Literal):
                                return key.schema

                            return key

                        additional_properties = additional_properties or _key_allows_additional_properties(key)
                        sub_schema = _to_schema(s[key], ignore_extra_keys=i)
                        key_name = _get_key_name(key)

                        if isinstance(key_name, str):
                            if not isinstance(key, Optional):
                                required_keys.append(key_name)
                            expanded_schema[key_name] = _json_schema(
                                sub_schema, is_main_schema=False, description=_get_key_description(key)
                            )
                            if isinstance(key, Optional) and hasattr(key, "default"):
                                expanded_schema[key_name]["default"] = _to_json_type(_invoke_with_optional_kwargs(key.default, **kwargs) if callable(key.default) else key.default)
                        elif isinstance(key_name, Or):
                            # JSON schema does not support having a key named one name or another, so we just add both options
                            # This is less strict because we cannot enforce that one or the other is required

                            for or_key in key_name.args:
                                expanded_schema[_get_key_name(or_key)] = _json_schema(
                                    sub_schema, is_main_schema=False, description=_get_key_description(or_key)
                                )

                    return_schema.update(
                        {
                            "type": "object",
                            "properties": expanded_schema,
                            "required": required_keys,
                            "additionalProperties": additional_properties,
                        }
                    )

            if is_main_schema:
                return_schema.update({"$id": schema_id, "$schema": "http://json-schema.org/draft-07/schema#"})
                if self._name:
                    return_schema["title"] = self._name

                if definitions_by_name:
                    return_schema["definitions"] = {}
                    for definition_name, definition in definitions_by_name.items():
                        return_schema["definitions"][definition_name] = definition

            return _create_or_use_ref(return_schema)

        return _json_schema(self, True)


class Optional(Schema):
    """Marker for an optional part of the validation Schema."""

    _MARKER = object()

    def __init__(self, *args, **kwargs):
        default = kwargs.pop("default", self._MARKER)
        super(Optional, self).__init__(*args, **kwargs)
        if default is not self._MARKER:
            # See if I can come up with a static key to use for myself:
            if _priority(self._schema) != COMPARABLE:
                raise TypeError(
                    "Optional keys with defaults must have simple, "
                    "predictable values, like literal strings or ints. "
                    '"%r" is too complex.' % (self._schema,)
                )
            self.default = default
            self.key = str(self._schema)

    def __hash__(self):
        return hash(self._schema)

    def __eq__(self, other):
        return (
            self.__class__ is other.__class__
            and getattr(self, "default", self._MARKER) == getattr(other, "default", self._MARKER)
            and self._schema == other._schema
        )

    def reset(self):
        if hasattr(self._schema, "reset"):
            self._schema.reset()


class Hook(Schema):
    def __init__(self, *args, **kwargs):
        self.handler = kwargs.pop("handler", lambda *args: None)
        super(Hook, self).__init__(*args, **kwargs)
        self.key = self._schema


class Forbidden(Hook):
    def __init__(self, *args, **kwargs):
        kwargs["handler"] = self._default_function
        super(Forbidden, self).__init__(*args, **kwargs)

    @staticmethod
    def _default_function(nkey, data, error):
        raise SchemaForbiddenKeyError("Forbidden key encountered: %r in %r" % (nkey, data), error)


class Literal(object):
    def __init__(self, value, description=None):
        self._schema = value
        self._description = description

    def __str__(self):
        return self._schema

    def __repr__(self):
        return 'Literal("' + self.schema + '", description="' + (self.description or "") + '")'

    @property
    def description(self):
        return self._description

    @property
    def schema(self):
        return self._schema


class Const(Schema):
    def validate(self, data, **kwargs):
        super(Const, self).validate(data, **kwargs)
        return data


def _callable_str(callable_):
    if hasattr(callable_, "__name__"):
        return callable_.__name__
    return str(callable_)


def _plural_s(sized):
    return "s" if len(sized) > 1 else ""
