import abc
import datetime
import subprocess
import zipfile
from abc import ABC
from typing import NoReturn, List
import re
import os
import sys
import shutil
from dateutil.relativedelta import relativedelta

import attr
from sentinelsat import SentinelAPI

from gis.geometry import lazy_property

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


def unzip_file_based_on_regex(file_location: str, regex: str, out_put_location: str) -> NoReturn:
    with zipfile.ZipFile(file_location, 'r') as myzip:
        for file in myzip.filelist:
            if re.match(regex, file.filename):
                myzip.extract(file, out_put_location)


def unzip_sentinel_data_based_on_key(key: str, regex: str):
    path = os.path.join(PATH_DATA_IN, key)
    try:
        zip_file = os.listdir(path)[0]
    except IndexError:
        raise FileNotFoundError("Seems that path is empty")

    zip_file_path = os.path.join(path, zip_file)
    out_put_path = os.path.join(PATH_DATA_OUT, key) + "_copy"
    if not os.path.exists(out_put_path):
        os.mkdir(out_put_path)
    unzip_file_based_on_regex(zip_file_path, regex, out_put_path)


def copy_path(from_path, to_path):
    shutil.copytree(from_path, to_path)


def move_the_data_and_remove(key: str):
    main_path = os.path.join(PATH_DATA_OUT, key) + "_copy"
    for path, dr, file in os.walk(main_path):
        if dr == ["R10m", "R20m", "R60m"]:
            for el in dr:
                copy_path(os.path.join(path, el), os.path.join(main_path[:-5], el))
    shutil.rmtree(main_path)


def move_the_data_full_process_based_on_key(key):
    to_path = os.path.join(PATH_DATA_OUT, key)
    if not os.path.exists(to_path):
        unzip_sentinel_data_based_on_key(key, r".*_B\d{1}\w{1}_\d{2}m.*")
        move_the_data_and_remove(key)
    else:
        try:
            shutil.rmtree(os.path.join(PATH_DATA_OUT, key) + "_copy")
        except FileNotFoundError:
            pass
        raise TypeError(f"Path {to_path} already exists")


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
class GdalScriptExec(CommandExecutorConcrete):

    """
    Remember to set system variable GDAL
    """
    files = attr.ib(type=list)
    output_location = attr.ib(type=str)

    def prepare_arguments(self):
        return [sys.executable,
                *super().prepare_arguments(),
                *self.files,
                "-o",
                self.output_location]


@attr.s
class GdalMerge(GdalScriptExec):
    COMMAND = os.path.join(
        check_for_sys_variable_existence("GDAL"),
        "swig\\python\\scripts\\gdal_merge.py"
    )


@attr.s
class User:
    name = attr.ib()
    password = attr.ib()


@attr.s
class SentinelDownloader:
    user = attr.ib(type=User)
    area = attr.ib(type=str)
    date = attr.ib(type=datetime.datetime)

    def __attrs_post_init__(self):
        self.month_range = relativedelta(months=1)
        self.api = SentinelAPI(
            self.user.name,
            self.user.password,
            'https://scihub.copernicus.eu/dhus'
        )

    def download_items(self):
        for scene in self.scenes[:1]:
            path = os.path.join(PATH_DATA_IN, scene)
            os.mkdir(path)
            self.api.download(scene, path)

    @property
    def scenes(self):
        end_positions = [[abs((self.items_metadata[key]["endposition"] - self.date)).total_seconds(), key]
                         for key in self.items_metadata]

        sorted_dates = sorted(end_positions, key=lambda x: x[0])

        small_difference = sorted_dates[0][0]

        final_dates = [el[1] for el in sorted_dates if el[0] == small_difference]

        return final_dates

    @lazy_property
    def items_metadata(self):
        return self.api.query(
            area=self.area,
            date=(self.date - self.month_range, self.date + self.month_range),
            platformname='Sentinel-2',
            cloudcoverpercentage="[0 TO 20]"
        )


SentinelDownloader(
    user=User(name="pawel"),

)