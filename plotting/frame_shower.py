@attr.s
class DataFrameShower:
    MARGIN = 2
    TRUNCATE_SIZE = 20
    data_frame = attr.ib(type=pandas.DataFrame)

    def show(self, limit=5, truncate=True):
        print(self.__create_string(limit, truncate))

    def __create_string(self, limit, truncate):
        columns_length = self.__find_max_length(limit, truncate)
        lines = list()
        lines.append(self.__prepare_row_string(self.data_frame.columns, columns_length))
        limited_data = self.data_frame.head(limit).values.tolist()

        for row in limited_data:
            lines.append(self.__prepare_row_string(row, columns_length))
        # dashes = "+" + "-" * (lines[0].__len__()-2) + "+"
        dashes = self.__create_dashes(columns_length)
        return dashes + "\n" + f"\n{dashes}\n".join(lines) + "\n" + dashes

    @staticmethod
    def __prepare_row_string(row, length_rows):
        rw = []
        for value, length in zip(row, length_rows):
            missing_length = length - len(str(value))
            if missing_length == 0:
                left_add = ""
                right_add = ""
            elif missing_length % 2 == 0:
                left_add = int(missing_length / 2) * " "
                right_add = int(missing_length / 2) * " "

            else:
                left_add = (int(missing_length / 2) + 1) * " "
                right_add = (int(missing_length / 2)) * " "

            rw.append(
                DataFrameShower.MARGIN * " " + DataFrameShower.MARGIN * " " + right_add + left_add + f"{str(value)[:length]}")

        return "|" + "|".join(rw) + "|"

    def __find_max_length(self, limit, truncate):
        maximums = []
        for col in self.data_frame.columns:
            max_length = self.data_frame.head(limit)[col].apply(lambda x: str(x).__len__()).max()
            max_length = max_length if max_length > len(col) else len(col)
            if truncate:
                maximums.append(min(max_length, self.TRUNCATE_SIZE) + self.MARGIN * 2)
            else:
                maximums.append(max_length + self.MARGIN * 2)
        return maximums

    def __create_dashes(self, columns_lengths: Iterable[int]):
        return "+" + "+".join(["-" * (length) + "-" * (DataFrameShower.MARGIN * 2) for length in columns_lengths]) + "+"
