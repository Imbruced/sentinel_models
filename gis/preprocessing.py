import abc
import subprocess
import zipfile
from abc import ABC
from typing import NoReturn, List
import re
import os
import sys

import attr

from gis import Extent

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
MODULE_PATH = os.path.split(SCRIPT_PATH)[0]

PATH_DATA_OUT = os.path.join(
    MODULE_PATH,
    "data/sentinel_out"
)

PATH_DATA_IN = os.path.join(
    MODULE_PATH,
    "data/sentinel_in"
)


def unzip_file_based_on_regex(file_location: str, regex: str, out_put_location: str = PATH_DATA_OUT) -> NoReturn:
    with zipfile.ZipFile(file_location, 'r') as myzip:
        for file in myzip.filelist:
            if re.match(regex, file.filename):
                myzip.extract(file, out_put_location)


def download_sentinel_data(extent: str, dates: tuple, cloud_filter: float, cloudcoverpercentage):
    pass


@attr.s
class CommandExecutor(ABC):
    COMMAND = ""
    arguments = attr.ib(type=List[str])

    @abc.abstractmethod
    def execute(self):
        """

        :return:
        """
        raise NotImplemented()

    @abc.abstractmethod
    def clean_args(self):
        """

        :return:
        """
        raise NotImplemented()

    @abc.abstractmethod
    def prepare_arguments(self):
        """

        :return:
        """
        raise NotImplemented()


@attr.s
class CommandExecutorConcrete(CommandExecutor):

    def execute(self):
        subprocess.run(self.clean_args,
                       stdout=subprocess.PIPE, shell=True)

    @property
    def clean_args(self):
        return self.prepare_arguments()

    def prepare_arguments(self):
        return [
            self.COMMAND,
            *self.arguments
        ]


def check_for_sys_variable_existence(variable: str) -> str:
    """
    takes the system variable name end returns the value for it, if variable does not exists
    it raise KeyError exception
    :param variable:
    :return: value for system variable
    """

    try:
        variable = os.environ[variable]
    except KeyError:
        raise KeyError(f"Variable {variable} does not exists")

    return variable


@attr.s
class GdalMerge(CommandExecutorConcrete):

    """
    Remember to set system variable GDAL
    """

    COMMAND = os.path.join(
        check_for_sys_variable_existence("GDAL"),
        "swig\\python\\scripts\\gdal_merge.py"
    )
    files = attr.ib(type=list)
    output_location = attr.ib(type=str)

    def prepare_arguments(self):
        return [sys.executable,
                *super().prepare_arguments(),
                *self.files,
                "-o",
                self.output_location]



