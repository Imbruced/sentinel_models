import attr
import gdal
import osr

from gis.gdal_image import GdalImage


@attr.s
class GeoTiffImageWriter:
    format_name = "geotiff"
    data = attr.ib()
    io_options = attr.ib()

    def save(self, path: str):
        drv = gdal.GetDriverByName("GTiff")
        band_number = self.data.shape[2]
        dtype = self.io_options["dtype"]

        ds = drv.Create(path, self.data.shape[1], self.data.shape[0], band_number, dtype)
        gdal_raster = GdalImage(ds, self.data.crs)
        srs = osr.SpatialReference()  # Establish its coordinate encoding
        srs.ImportFromEPSG(self.data.crs.code)
        transformed_ds = gdal_raster.transform(self.data.extent.origin, self.data.pixel, srs.ExportToWkt())

        for band in range(self.data.shape[2]):
            transformed_ds.ds.GetRasterBand(band + 1).WriteArray(self.data[:, :, band])
