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

    src.annotations.param.py
    ~~~~~~~~~~~~~~~~~~~~~~~~

"""
import logging
import json
import os

from sppas import paths
from sppas import annots

from sppas.src.config import msg
from sppas.src.structs.baseoption import sppasOption
from sppas.src.structs.lang import sppasLangResource
from sppas.src.structs.lang import UNDETERMINED
from sppas.src.anndata.aio import extensions as annots_ext
from sppas.src.audiodata.aio import extensions as audio_ext
from sppas.src.utils.fileutils import sppasDirUtils
from sppas.src.utils.fileutils import sppasFileUtils

# ----------------------------------------------------------------------------


class annotationParam(object):
    """Annotation parameters data manager.

    :author:       Brigitte Bigi
    :organization: Laboratoire Parole et Langage, Aix-en-Provence, France
    :contact:      develop@sppas.org
    :license:      GPL, v3
    :copyright:    Copyright (C) 2011-2018  Brigitte Bigi

    Class to store data of an automatic annotation like its name, description,
    supported languages, etc.

    """

    def __init__(self, filename=None):
        """Create a new annotationParam instance.

        :param filename: (str) Annotation configuration file

        """
        # An identifier to represent this annotation
        self.__key = None
        # The name of the annotation
        self.__name = ""
        # The description of the annotation
        self.__descr = ""
        # The status of the annotation
        self.__enabled = False
        self.__invalid = False
        # The language resource: sppasLangResource()
        self.__resources = list()
        # The list of options
        self.__options = list()

        # Fix all members from a given config file
        if filename is not None:
            self.parse(filename)

    # ------------------------------------------------------------------------

    def parse(self, filename):
        """Parse a configuration file to fill members.

        :param filename: (str) Annotation configuration file (.ini)

        """
        if filename.endswith('.json'):

            config = os.path.join(paths.etc, filename)
            if os.path.exists(config) is False:
                raise IOError('Installation error: the file to configure the '
                              'automatic annotations does not exist.')

            # Read the whole file content
            with open(config) as cfg:
                conf = json.load(cfg)

            self.__key = conf['id']
            self.__name = msg(conf.get('name', ''), "annotations)")  # translate the name
            self.__descr = conf.get('descr', "")

            for new_option in conf['options']:
                opt = sppasOption(new_option['id'])
                opt.set_type(new_option['type'])
                opt.set_value(str(new_option['value']))  # dangerous cast
                opt.set_text(msg(new_option.get('text', ''), "annotations"))   # translated
                self.__options.append(opt)

            for new_resource in conf['resources']:
                lr = sppasLangResource()
                lr.set(new_resource['type'],
                       new_resource['path'],
                       new_resource.get('name', ''),
                       new_resource['ext'])
                self.__resources.append(lr)

        else:
            raise IOError('Unknown extension for filename {:s}'.format(filename))

    # -----------------------------------------------------------------------
    # Setters
    # -----------------------------------------------------------------------

    def set_activate(self, activate):
        """Enable the annotation but only if this annotation is valid.

        :param activate: (bool) Enable or disable the annotation
        :returns: (bool) enabled or disabled

        """
        self.__enabled = activate
        if activate is True and self.__invalid is True:
            self.__enabled = False
        return self.__enabled

    # -----------------------------------------------------------------------

    def set_lang(self, lang):
        """Set the language of the annotation, if this latter is accepted.

        :param lang: (str) Language to fix for the annotation
        :returns: (bool) Language is set or not

        """
        if len(self.__resources) > 0:
            try:
                self.__resources[0].set_lang(lang)
                return True
            except:
                self.__invalid = True
                self.__enabled = False
                return False
        return True

    # -----------------------------------------------------------------------
    # Getters
    # -----------------------------------------------------------------------

    def get_key(self):
        """Return the identifier of the annotation (str)."""
        return self.__key

    # -----------------------------------------------------------------------

    def get_name(self):
        """Return the name of the annotation (str)."""
        return self.__name

    # -----------------------------------------------------------------------

    def get_descr(self):
        """Return the description of the annotation (str)."""
        return self.__descr

    # -----------------------------------------------------------------------

    def get_activate(self):
        """Return the activation status of the annotation (bool)."""
        return self.__enabled

    # -----------------------------------------------------------------------

    def get_lang(self):
        """Return the language or an empty string."""
        if len(self.__resources) > 0:
            return self.__resources[0].get_lang()
        return ""

    # -----------------------------------------------------------------------

    def get_langlist(self):
        """Return the list of available languages (list of str)."""
        if len(self.__resources) > 0:
            return self.__resources[0].get_langlist()
        return []

    # -----------------------------------------------------------------------

    def get_langresource(self):
        """Return the list of language resources."""
        return [r.get_langresource() for r in self.__resources]

    # -----------------------------------------------------------------------

    def get_options(self):
        """Return the list of options of the annotation."""
        return self.__options

    # -----------------------------------------------------------------------

    def get_option(self, step):
        """Return the step-th option."""
        return self.__options[step]

    # -----------------------------------------------------------------------

    def get_option_by_key(self, key):
        """Return an option from its key."""
        for opt in self.__options:
            if key == opt.get_key():
                return opt

    # -----------------------------------------------------------------------

    def set_option_value(self, key, value):
        """Change value of an option.

        :param key: (str) Identifier of the option
        :param value: (any) New value for the option
        :raises: KeyError

        """
        # the option is already in the list, change its value
        for opt in self.__options:
            if key == opt.get_key():
                opt.set_value(value)
                return

        # the option was not found in the list
        raise KeyError("Unknown option {:s} in annotation parameters."
                       "".format(key))

# ---------------------------------------------------------------------------


class sppasParam(object):
    """Annotation parameters manager.

    :author:       Brigitte Bigi
    :organization: Laboratoire Parole et Langage, Aix-en-Provence, France
    :contact:      develop@sppas.org
    :license:      GPL, v3
    :copyright:    Copyright (C) 2011-2018  Brigitte Bigi

    Parameters of a set of annotations.

    """

    def __init__(self, annotation_keys=None):
        """Create a new sppasParam instance with default values.

        :param annotation_keys: (list) List of annotations to load. None=ALL.

        """
        # A log file to communicate to the user
        self._report = None

        # The format of the annotated files
        self._output_ext = annots.extension

        # Input files to annotate
        self._inputs = []

        # The parameters of all the annotations
        self.annotations = []
        self.load_annotations(annotation_keys)

    # ------------------------------------------------------------------------

    def load_annotations(self, annotation_files=None):
        """Load the annotation configuration files.

        Load from a list of given file names (without path) or from the
        default sppas ui configuration file.

        :param annotation_files: (list) List of annotations to load. None=ALL.

        """
        if not annotation_files or len(annotation_files) == 0:
            self.parse_config_file()

        else:
            for cfg_file in annotation_files:
                self.__load(os.path.join(paths.etc, cfg_file))

    # ------------------------------------------------------------------------

    def parse_config_file(self):
        """Parse the sppasui.json file.

        Parse the file to get the list of annotations and parse the
        corresponding "ini" file.

        """
        config = os.path.join(paths.etc, "sppasui.json")
        if os.path.exists(config) is False:
            raise IOError('Installation error: the file to configure the '
                          'automatic annotations does not exist.')

        # Read the whole file content
        with open(config) as cfg:
            dict_cfg = json.load(cfg)

        # Load annotation configurations
        for ann in dict_cfg["annotate"]:
            if ann["gui"] is True:
                self.__load(os.path.join(paths.etc, ann["config"]))

    # -----------------------------------------------------------------------

    def __load(self, cfg_file):
        """Load parameters of an annotation from its configuration file."""
        try:
            a = annotationParam(cfg_file)
            self.annotations.append(a)
        except:
            logging.error('Configuration file {:s} not loaded.'
                          ''.format(cfg_file))

    # -----------------------------------------------------------------------
    # Input entries to annotate
    # -----------------------------------------------------------------------

    def get_sppasinput(self):
        """Return the list of entries to annotate."""
        return self._inputs

    # -----------------------------------------------------------------------

    def add_sppasinput(self, entry):
        """Add a new entry to annotate.

        If no report file name was previously fixed, it will be assigned to
        the given entry with the extension '.log'.

        :param entry: (str) Filename or directory

        """
        base_ext = ['.txt', '.hz', '.pitchtier']
        for e in audio_ext:
            base_ext.append(e.lower())
        ann_ext = [e.lower() for e in annots_ext]

        # Create a full list of files to add, without any kind of filter
        initial_files = list()
        if os.path.isdir(entry):
            # Get the list of files from the input directory
            for ae in base_ext:
                initial_files.extend(sppasDirUtils(entry).get_files(ae))
        else:
            initial_files.append(entry)

        # Create a list with the basename of the files
        for entry_file in initial_files:
            fn, e = os.path.splitext(entry_file)
            if fn in self._inputs:
                continue

            if len(e) == 0:
                # the entry is already the basename
                self.__append_input(fn, base_ext)

            elif e.lower() in base_ext:
                # the entry is a primary file (audio/text/pitch)
                self._inputs.append(fn)

            elif e.lower() in ann_ext:
                # the entry is an annotated file
                appended = self.__append_input(fn, base_ext)
                if appended is False:
                    # the entry is an annotated file with a pattern
                    fn = self.__remove_pattern(fn)
                    if fn not in self._inputs:
                        self.__append_input(fn, base_ext)

        for f in self._inputs:
            logging.debug(f)

    # -----------------------------------------------------------------------

    def __append_input(self, base_fn, base_ext):
        for ae in base_ext:
            if sppasFileUtils(base_fn + ae).exists() \
                    and base_fn not in self._inputs:
                self._inputs.append(base_fn)
                return True
        return False

    # -----------------------------------------------------------------------

    @staticmethod
    def __remove_pattern(entry):
        minus = entry.rfind('-')
        if minus != -1:
            sep = entry.rfind(os.path.sep)
            if minus > sep:
                return entry[:minus]
        return entry

    # -----------------------------------------------------------------------
    # Procedure Outcome Report file name
    # -----------------------------------------------------------------------

    def set_report_filename(self, filename):
        """Fix the name of the file to save the report of the annotations.

        :param filename: (str) Filename for the Procedure Outcome Report

        """
        self._report = filename

    # -----------------------------------------------------------------------

    def get_report_filename(self):
        """Return the name of the file for the Procedure Outcome Report."""
        return self._report

    # -----------------------------------------------------------------------
    # selected language
    # -----------------------------------------------------------------------

    def set_lang(self, language, step=None):
        if step is not None:
            self.annotations[step].set_lang(language)
        else:
            for a in self.annotations:
                a.set_lang(language)

    def get_lang(self, step=None):
        if step is None:
            for a in self.annotations:
                if a.get_lang() != UNDETERMINED:
                    return a.get_lang()
            return UNDETERMINED
        return self.annotations[step].get_lang()

    def get_langresource(self, step):
        return self.annotations[step].get_langresource()

    # ------------------------------------------------------------------------
    # annotations
    # ------------------------------------------------------------------------

    def activate_annotation(self, stepname):
        for i, a in enumerate(self.annotations):
            if a.get_key() == stepname:
                a.set_activate(True)
                return i
        return -1

    def activate_step(self, step):
        self.annotations[step].set_activate(True)

    def disable_step(self, step):
        self.annotations[step].set_activate(False)

    def get_step_status(self, step):
        return self.annotations[step].get_activate()

    def get_step_name(self, step):
        return self.annotations[step].get_name()

    def get_step_descr(self, step):
        return self.annotations[step].get_descr()

    # ------------------------------------------------------------------------

    def get_step_idx(self, annotation_key):
        """Get the annotation step index from an annotation key.

        :param annotation_key: (str)
        :raises: KeyError

        """
        for i, a in enumerate(self.annotations):
            if a.get_key() == annotation_key:
                return i

        raise KeyError('No configuration file is available for an annotation'
                       'with key {:s}'.format(annotation_key))

    # ------------------------------------------------------------------------

    def get_step_key(self, step):
        return self.annotations[step].get_key()

    def get_step_numbers(self):
        return len(self.annotations)

    def get_steplist(self):
        steps = []
        for i in range(len(self.annotations)):
            steps.append(self.annotations[i].get_name())
        return steps

    def get_langlist(self, step=2):
        return self.annotations[step].get_langlist()

    def get_step(self, step):
        """Return the 'sppasParam' instance of the annotation."""
        return self.annotations[step]

    def get_options(self, step):
        return self.annotations[step].get_options()

    def set_option_value(self, step, key, value):
        self.annotations[step].set_option_value(key, value)

    # -----------------------------------------------------------------------
    # Annotation file output format
    # -----------------------------------------------------------------------

    def get_output_format(self):
        """Return the output format of the annotations (extension)."""
        return self._output_ext

    # -----------------------------------------------------------------------

    def set_output_format(self, output_format):
        """Fix the output format of the annotations.

        :param output_format: (str) File extension (with or without a dot)
        :returns: the extension really set.

        """
        # Force to contain the dot
        if not output_format.startswith("."):
            output_format = "." + output_format

        # Force to use the appropriate upper-lower cases
        for e in annots_ext:
            if output_format.lower() == e.lower():
                output_format = e

        # Check if this extension is know. If not, set to the default.
        if output_format not in annots_ext:
            # Instead we could raise an exception...
            logging.warning(
                "Unknown extension: {:s}. Output format is set to the "
                "default: {:s}.".format(output_format, annots.extension))
            output_format = annots.extension

        self._output_ext = output_format
        return output_format
