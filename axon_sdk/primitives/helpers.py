# Axon SDK — A simulation framework for spike-timing-based neural computation using the STICK model.
# Copyright (C) 2024–2025 Neucom ApS
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Utilities
=========

Common helper functions used throughout the Axon SDK.
"""

def flatten_nested_list(nested_list: list) -> list:
    """
        Recursively flattens an arbitrarily nested list into a single flat list.

        This function supports arbitrary nesting of Python lists and returns a
        single list containing all the elements in depth-first order.

        Args:
            nested_list (list): A list that may contain other lists as elements.

        Returns:
            list: A flat list with all nested elements extracted.

        Example:
            >>> flatten_nested_list([1, [2, [3, 4], 5]])
            [1, 2, 3, 4, 5]
    """
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_nested_list(item))
        else:
            flat_list.append(item)
    return flat_list
