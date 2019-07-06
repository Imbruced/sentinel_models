from unittest import TestCase

from gis.preprocessing import CommandExecutor, GdalMerge, check_for_sys_variable_existence, CommandExecutorConcrete


class TestPreprocessing(TestCase):

    def test_command_executor_instance(self):
        exec = CommandExecutorConcrete(arguments={"a": ""})

    def test_command_creation(self):
        gm = GdalMerge(arguments=["-ps", "10",  "10"], files=["file1"], output_location="file_output")
        self.assertEqual(gm.clean_args, ["C:\\Users\\Pawel\\Anaconda3\\envs\\tensorflow_env\\python.exe",
                                         "C:\\Program Files (x86)\\gdal-2.4.1\\swig\\python\\scripts\\gdal_merge.py",
                                         "-ps",
                                         "10",
                                         "10",
                                         "file1",
                                         "-o",
                                         "file_output"])

    def test_sys_executable_variable_taking(self):
        check_for_sys_variable_existence(
            "windir"
        )

    def test_wrong_system_variable(self):
        with self.assertRaises(KeyError):
            check_for_sys_variable_existence(
            "random_variable_which_does_not_exists"
        )

    def test_command_executing(self):
        gm = GdalMerge(
            arguments=["-ps", "10",  "10", "-separate"],
            files=["file 1"],
            output_location="file2"
        )
        print(gm.clean_args)

    def test_gdal_merge_executoing(self):
        gm = GdalMerge(
            arguments=["-ps", "10",  "10", "-separate"],
            files=["D:\\master_thesis\\data\\20160629\\img\\T34UFB_20160629T094032_B09.jp2",
                   "D:\\master_thesis\\data\\20160629\\img\\T34UFB_20160629T094032_B10.jp2"],
            output_location="D:\\master_thesis\\data\\20160629\\img\\merged_image.tif"
        )
        gm.execute()
