class PointFrame(GeometryFrame):

    def __init__(self, frame, geometry_column):
        super().__init__(frame, geometry_column)
        super().__attr__post_init__()


class LineFrame(GeometryFrame):
    def __init__(self, frame: gpd.GeoDataFrame, geometry_column: str):
        super().__init__(frame, geometry_column)
        super().__attr__post_init__()


class PolygonFrame(GeometryFrame):
    def __init__(self, frame: gpd.GeoDataFrame, geometry_column: str):
        super().__init__(frame, geometry_column)
        super().__attr__post_init__()
