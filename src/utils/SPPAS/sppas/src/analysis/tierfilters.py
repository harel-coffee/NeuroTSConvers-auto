# -*- coding: UTF-8 -*-
"""
    ..
        ---------------------------------------------------------------------
         ___   __    __    __    ___
        /     |  \  |  \  |  \  /              the automatic
        \__   |__/  |__/  |___| \__             annotation and
           \  |     |     |   |    \             analysis
        ___/  |     |     |   | ___/              of speech

        http://www.sppas.org/

        Use of this software is governed by the GNU Public License, version 3.

        SPPAS is free software: you can redistribute it and/or modify
        it under the terms of the GNU General Public License as published by
        the Free Software Foundation, either version 3 of the License, or
        (at your option) any later version.

        SPPAS is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU General Public License for more details.

        You should have received a copy of the GNU General Public License
        along with SPPAS. If not, see <http://www.gnu.org/licenses/>.

        This banner notice must not be removed.

        ---------------------------------------------------------------------

    anndata.filter.filters.py
    ~~~~~~~~~~~~~~~~~~~~~~~~~

"""

from sppas.src.anndata.anndataexc import AnnDataValueError
from sppas.src.anndata.anndataexc import AnnDataKeyError

from sppas.src.utils.makeunicode import u

from sppas.src.anndata.ann.annset import sppasAnnSet
from sppas.src.anndata.ann.annlabel import sppasTagCompare
from sppas.src.anndata.ann.annlocation import sppasDurationCompare
from sppas.src.anndata.ann.annlocation import sppasLocalizationCompare
from sppas.src.anndata.ann.annlocation import sppasIntervalCompare

# ---------------------------------------------------------------------------


class sppasTierFilters(object):
    """This class implements the 'SPPAS tier' filter system.

    :author:       Brigitte Bigi
    :organization: Laboratoire Parole et Langage, Aix-en-Provence, France
    :contact:      develop@sppas.org
    :license:      GPL, v3
    :copyright:    Copyright (C) 2011-2018  Brigitte Bigi

    Search in tiers. The class sppasTierFilters() allows to create several types
    of filter (tag, duration, ...), and the class sppasAnnSet() is a data set
    manager, i.e. it contains the annotations selected by a filter and a
    string representing the filter.

    Create a filter:

        >>> f = sppasTierFilters(tier)

    then, apply a filter with some pattern like in the following examples.
    sppasAnnSet() can be combined with operators & and |, like for any other
    'set' in Python, 'an unordered collection of distinct hashable objects'.

    :Example1: extract silences:

        >>> f.tag(exact=u('#')))

    :Example2: extract silences more than 200ms

        >>> f.tag(exact=u("#")) & f.dur(gt=0.2)

    :Example3: find the annotations with at least a label with a tag
    starting by "pa" and ending by "a" like "pa", "papa", "pasta", etc:

        >>> f.tag(startswith="pa", endswith='a')

    It's equivalent to write:

        >>> f.tag(startswith="pa", endswith='a', logic_bool="and")

    The classical "and" and "or" logical boolean predicates are accepted;
    "and" is the default one. It defines whether all the functions must
    be True ("and") or any of them ("or").

    The result of the two previous lines of code is the same, but two
    times faster, compared to use this one:

        >>> f.tag(startswith="pa") & f.tag(endswith='a')

    In the first case, for each tag, the method applies the logical boolean
    between two predicates and creates the data set matching the combined
    condition. In the second case, each call to the method creates a data
    set matching each individual condition, then the data sets are
    combined.

    """

    def __init__(self, tier):
        """Create a sppasTierFilters instance.

        :param tier: (sppasTier) The tier to be filtered.

        """
        self.tier = tier

    # -----------------------------------------------------------------------

    def tag(self, **kwargs):
        """Apply functions on all tags of all labels of annotations.

        Each argument is made of a function name and its expected value.
        Each function can be prefixed with 'not_', like in the next example.

        :Example:

            >>> f.tag(startswith="pa", not_endswith='a', logic_bool="and")
            >>> f.tag(startswith="pa") & f.tag(not_endswith='a')
            >>> f.tag(startswith="pa") | f.tag(startswith="ta")

        :param kwargs: logic_bool/any sppasTagCompare() method.
        :returns: (sppasAnnSet)

        """
        comparator = sppasTagCompare()

        # extract the information from the arguments
        sppasTierFilters.__test_args(comparator, **kwargs)
        logic_bool = sppasTierFilters.__fix_logic_bool(**kwargs)
        tag_fct_values = sppasTierFilters.__fix_function_values(comparator,
                                                            **kwargs)
        tag_functions = sppasTierFilters.__fix_functions(comparator,
                                                     **kwargs)

        data = sppasAnnSet()

        # search the annotations to be returned:
        for annotation in self.tier:

            # any label can match
            for label in annotation.get_labels():

                is_matching = label.match(tag_functions, logic_bool)
                # no need to test the next labels if the current one is matching.
                if is_matching is True:
                    data.append(annotation, tag_fct_values)
                    break

        return data

    # -----------------------------------------------------------------------

    def dur(self, **kwargs):
        """Apply functions on durations of the location of annotations.

        :param kwargs: logic_bool/any sppasDurationCompare() method.
        :returns: (sppasAnnSet)

        Examples:
            >>> f.dur(ge=0.03) & f.dur(le=0.07)
            >>> f.dur(ge=0.03, le=0.07, logic_bool="and")

        """
        comparator = sppasDurationCompare()

        # extract the information from the arguments
        sppasTierFilters.__test_args(comparator, **kwargs)
        logic_bool = sppasTierFilters.__fix_logic_bool(**kwargs)
        dur_fct_values = sppasTierFilters.__fix_function_values(comparator,
                                                            **kwargs)
        dur_functions = sppasTierFilters.__fix_functions(comparator,
                                                     **kwargs)

        data = sppasAnnSet()

        # search the annotations to be returned:
        for annotation in self.tier:

            location = annotation.get_location()
            is_matching = location.match_duration(dur_functions, logic_bool)
            if is_matching is True:
                data.append(annotation, dur_fct_values)

        return data

    # -----------------------------------------------------------------------

    def loc(self, **kwargs):
        """Apply functions on localizations of annotations.

        :param kwargs: logic_bool/any sppasLocalizationCompare() method.
        :returns: (sppasAnnSet)

        :Example:

            >>> f.loc(rangefrom=3.01) & f.loc(rangeto=10.07)
            >>> f.loc(rangefrom=3.01, rangeto=10.07, logic_bool="and")

        """
        comparator = sppasLocalizationCompare()

        # extract the information from the arguments
        sppasTierFilters.__test_args(comparator, **kwargs)
        logic_bool = sppasTierFilters.__fix_logic_bool(**kwargs)
        loc_fct_values = sppasTierFilters.__fix_function_values(comparator, **kwargs)
        loc_functions = sppasTierFilters.__fix_functions(comparator, **kwargs)

        data = sppasAnnSet()

        # search the annotations to be returned:
        for annotation in self.tier:

            location = annotation.get_location()
            is_matching = location.match_localization(loc_functions, logic_bool)
            if is_matching is True:
                data.append(annotation, loc_fct_values)

        return data

    # -----------------------------------------------------------------------

    def rel(self, other_tier, *args, **kwargs):
        """Apply functions of relations between localizations of annotations.

        :param other_tier: the tier to be in relation with.
        :param args: any sppasIntervalCompare() method.
        :param kwargs: any option of the methods.
        :returns: (sppasAnnSet)

        :Example:

            >>> f.rel(other_tier, "equals",
            >>>                   "overlaps",
            >>>                   "overlappedby", min_overlap=0.04)

        """
        comparator = sppasIntervalCompare()

        # extract the information from the arguments
        rel_functions = sppasTierFilters.__fix_relation_functions(comparator,
                                                              *args)

        data = sppasAnnSet()

        # search the annotations to be returned:
        for annotation in self.tier:

            location = annotation.get_location()
            match_values = sppasTierFilters.__connect(location,
                                                  other_tier,
                                                  rel_functions,
                                                  **kwargs)
            if len(match_values) > 0:
                data.append(annotation, list(set(match_values)))

        return data

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    @staticmethod
    def cast_data(tier, sfilter, entry):
        """Return an entry into the appropriate type.

        :param tier: (sppasTier)
        :param sfilter: (str) Name of the filter (tag, loc, ...)
        :param entry: (str) The entry to cast
        :return: typed entry

        """
        if sfilter == "tag":
            if tier.is_float():
                return float(entry)
            elif tier.is_int():
                return int(entry)
            elif tier.is_bool():
                return bool(entry)

        elif sfilter == "loc":
            p = tier.get_first_point()
            if isinstance(p.get_midpoint(), int):
                return int(entry)
            else:
                return float(entry)

        elif sfilter == "dur":
            return float(entry)

        return u(entry)

    # -----------------------------------------------------------------------
    # Private
    # -----------------------------------------------------------------------

    @staticmethod
    def __test_args(comparator, **kwargs):
        """Raise an exception if any of the args is not correct."""

        names = ["logic_bool"] + comparator.get_function_names()
        for func_name, value in kwargs.items():
            if func_name.startswith("not_"):
                func_name = func_name[4:]

            if func_name not in names:
                raise AnnDataKeyError("kwargs function name", func_name)

    # -----------------------------------------------------------------------

    @staticmethod
    def __fix_logic_bool(**kwargs):
        """Return the value of a logic boolean predicate."""

        for func_name, value in kwargs.items():
            if func_name == "logic_bool":
                if value not in ['and', 'or']:
                    raise AnnDataValueError(value, "logic bool")
                return value
        return "and"

    # -----------------------------------------------------------------------

    @staticmethod
    def __fix_function_values(comparator, **kwargs):
        """Return the list of function names and the expected value."""

        fct_values = list()
        for func_name, value in kwargs.items():
            if func_name in comparator.get_function_names():
                fct_values.append("{:s} = {!s:s}".format(func_name, value))

        return fct_values

    # -----------------------------------------------------------------------

    @staticmethod
    def __fix_functions(comparator, **kwargs):
        """Parse the args to get the list of function/value/complement."""

        f_functions = list()
        for func_name, value in kwargs.items():

            logical_not = False
            if func_name.startswith("not_"):
                logical_not = True
                func_name = func_name[4:]

            if func_name in comparator.get_function_names():
                f_functions.append((comparator.get(func_name),
                                    value,
                                    logical_not))

        return f_functions

    # -----------------------------------------------------------------------

    @staticmethod
    def __fix_relation_functions(comparator, *args):
        """Parse the arguments to get the list of function/complement."""

        f_functions = list()
        for func_name in args:

            logical_not = False
            if func_name.startswith("not_"):
                logical_not = True
                func_name = func_name[4:]

            if func_name in comparator.get_function_names():
                f_functions.append((comparator.get(func_name),
                                    logical_not))
            else:
                raise AnnDataKeyError("args function name", func_name)

        return f_functions

    # -----------------------------------------------------------------------

    @staticmethod
    def __connect(location, other_tier, rel_functions, **kwargs):
        """Find connections between location and the other tier."""

        values = list()
        for other_ann in other_tier:
            for localization, score in location:
                for other_loc, other_score in other_ann.get_location():
                    for func_name, complement in rel_functions:
                        is_connected = func_name(localization,
                                                 other_loc,
                                                 **kwargs)
                        if is_connected:
                            values.append(func_name.__name__)

        return values
