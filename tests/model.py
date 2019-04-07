import matplotlib
from gis.geometry import PolygonFrame
from gis.raster import Raster
import matplotlib.pyplot as plt
from gis.log_lib import logger
from gis.raster import RasterData

matplotlib.use('Qt5Agg')

if __name__ == "__main__":
    shape_path = "D:\\master_thesis\\PYTHON\\sentinel_models\\tests\\domy.shp"
    polygon_frame = PolygonFrame.from_file(shape_path).union("id")
    logger.info(polygon_frame.crs)
    main_image = Raster.from_file("D:\\master_thesis\\PYTHON\\sentinel_models\\tests\\buildings.tif", crs="epsg:26917")
    label_raster = Raster.with_adjustment("from_geo", main_image, polygon_frame)
    raster_data = RasterData(main_image, label_raster)
    plt.imshow(label_raster.array[:, :, 0])
    plt.show()
    raster_data.save_unet_images([128, 128], "C:\\Users\\Pawel\\Desktop\\sentinel_models\\tests")

