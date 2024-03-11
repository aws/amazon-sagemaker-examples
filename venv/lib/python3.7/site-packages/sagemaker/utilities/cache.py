# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""This module defines a LRU cache class."""
from __future__ import absolute_import

import datetime
import collections
from typing import TypeVar, Generic, Callable, Optional

KeyType = TypeVar("KeyType")
ValType = TypeVar("ValType")


class LRUCache(Generic[KeyType, ValType]):
    """Class that implements LRU cache with expiring items.

    LRU caches remove items in a FIFO manner, such that the oldest
    items to be used are the first to be removed.
    If you attempt to retrieve a cache item that is older than the
    expiration time, the item will be invalidated.
    """

    class Element:
        """Class describes the values in the cache.

        This object stores the value itself as well as a timestamp so that this
        element can be invalidated if it becomes too old.
        """

        def __init__(self, value: ValType, creation_time: datetime.datetime):
            """Initialize an ``Element`` instance for ``LRUCache``.

            Args:
                value (ValType): Value that is stored in cache.
                creation_time (datetime.datetime): Time at which cache item was created.
            """
            self.value = value
            self.creation_time = creation_time

    def __init__(
        self,
        max_cache_items: int,
        expiration_horizon: datetime.timedelta,
        retrieval_function: Callable[[KeyType, ValType], ValType],
    ) -> None:
        """Initialize an ``LRUCache`` instance.

        Args:
            max_cache_items (int): Maximum number of items to store in cache.
            expiration_horizon (datetime.timedelta): Maximum time duration a cache element can
                persist before being invalidated.
            retrieval_function (Callable[[KeyType, ValType], ValType]): Function which maps cache
                keys and current values to new values. This function must have kwarg arguments
                ``key`` and ``value``. This function is called as a fallback when the key
                is not found in the cache, or a key has expired.

        """
        self._max_cache_items = max_cache_items
        self._lru_cache: collections.OrderedDict = collections.OrderedDict()
        self._expiration_horizon = expiration_horizon
        self._retrieval_function = retrieval_function

    def __len__(self) -> int:
        """Returns number of elements in cache."""
        return len(self._lru_cache)

    def __contains__(self, key: KeyType) -> bool:
        """Returns True if key is found in cache, False otherwise.

        Args:
            key (KeyType): Key in cache to retrieve.
        """
        return key in self._lru_cache

    def clear(self) -> None:
        """Deletes all elements from the cache."""
        self._lru_cache.clear()

    def get(self, key: KeyType, data_source_fallback: Optional[bool] = True) -> ValType:
        """Returns value corresponding to key in cache.

        Args:
            key (KeyType): Key in cache to retrieve.
            data_source_fallback (Optional[bool]): True if data should be retrieved if
                it's stale or not in cache. Default: True.
            Raises:
                KeyError: If key is not found in cache or is outdated and
                ``data_source_fallback`` is False.
        """
        if data_source_fallback:
            if key in self._lru_cache:
                return self._get_item(key, False)
            self.put(key)
            return self._get_item(key, False)
        return self._get_item(key, True)

    def put(self, key: KeyType, value: Optional[ValType] = None) -> None:
        """Adds key to cache using ``retrieval_function``.

        If value is provided, this is used instead. If the key is already in cache,
        the old element is removed. If the cache size exceeds the size limit, old
        elements are removed in order to meet the limit.

        Args:
            key (KeyType): Key in cache to retrieve.
            value (Optional[ValType]): Value to store for key. Default: None.
        """
        curr_value = None
        if key in self._lru_cache:
            curr_value = self._lru_cache.pop(key)

        while len(self._lru_cache) >= self._max_cache_items:
            self._lru_cache.popitem(last=False)

        if value is None:
            value = self._retrieval_function(  # type: ignore
                key=key, value=curr_value.element if curr_value else None
            )

        self._lru_cache[key] = self.Element(
            value=value, creation_time=datetime.datetime.now(tz=datetime.timezone.utc)
        )

    def _get_item(self, key: KeyType, fail_on_old_value: bool) -> ValType:
        """Returns value from cache corresponding to key.

        If ``fail_on_old_value``, a KeyError is raised instead of a new value
            getting fetched.

        Args:
            key (KeyType): Key in cache to retrieve.
            fail_on_old_value (bool): True if a KeyError is raised when the cache value
                is old.

        Raises:
            KeyError: If key is not in cache or if key is old in cache
                and fail_on_old_value is True.
        """
        try:
            element = self._lru_cache.pop(key)
            curr_time = datetime.datetime.now(tz=datetime.timezone.utc)
            element_age = curr_time - element.creation_time
            if element_age > self._expiration_horizon:
                if fail_on_old_value:
                    raise KeyError(
                        f"{key} has aged beyond allowed time {self._expiration_horizon}. "
                        f"Element created at {element.creation_time}."
                    )
                element.value = self._retrieval_function(  # type: ignore
                    key=key, value=element.value
                )
                element.creation_time = curr_time
            self._lru_cache[key] = element
            return element.value
        except KeyError:
            raise KeyError(f"{key} not found in LRUCache!")
