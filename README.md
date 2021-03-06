# Sentinel models
This library was created to create input data for Neural Network based on
Satellite images data. First use was on setninel2 satellite images but it 
will also work for other data sources. Library is in beta version still,
some of the functionality may not be available. 
## Requirements
## Use cases
<h2> Class Raster</h2>

 This class allows to load data from various formats. Currently available are:
 <li> Shapefile (shp) </li>
 <li> Geotiff (geotiff) </li>
 <li> PNG (png) </li>
 <li> WKT (wkt) </li>
 <br>
 When the shape file or any other geometry format is loaded, it is converted into 
 raster based on coordinates, Array size depends on pixel size and extent size.
 It can be passed via extent and pixel in options method.
 
 <h4> Loading from wkt</h4>
 
```python
wkt = "Polygon((110.0 105.0, 110.0 120.0, 120.0 120.0, 120.0 110.0, 110.0 105.0))"
raster = Raster\
            .read\
            .format("wkt")\
            .load(wkt)
raster.show()
```
<img src="https://github.com/Imbruced/sentinel_models/blob/raster_refactor/docs/images/raster_from_wkt_default_parameters.PNG" width="200">

 <h4> Loading from Shape File</h4>
 

```python
shape_path = "/tests/data/shapes/domy.shp"

GeometryFrame.from_file(shape_path).show()
"""
 -----------------------------------------------------------------------
|      id     |     cls     |                 geometry                 |
------------------------------------------------------------------------
|      1      |      4      |   POLYGON ((559610.9241914469 4929878.   |
------------------------------------------------------------------------
|      1      |      5      |   POLYGON ((559649.5178541846 4929868.   |
------------------------------------------------------------------------
|      1      |      5      |   POLYGON ((559678.1483190297 4929851.   |
------------------------------------------------------------------------
|      1      |      2      |   POLYGON ((559648.8379760745 4929819.   |
------------------------------------------------------------------------
|      1      |      4      |   POLYGON ((559587.8988370569 4929857.   |
------------------------------------------------------------------------
"""
raster = Raster\
            .read\
            .format("shp")\
            .load(shape_path)
raster.show()
```

<img src="https://github.com/Imbruced/sentinel_models/blob/raster_refactor/docs/images/raster_from_shp_default_parameters.PNG" width="250">

To get raster value based on shape file column, use color_column attribute in options

```python
raster = Raster\
            .read\
            .format("shp")\
            .options(
                pixel=Pixel(0.5, 0.5),
                color_column="cls"
            ).load(shape_path)
raster.show()
```
<img src="https://github.com/Imbruced/sentinel_models/blob/raster_refactor/docs/images/raster_from_shp_pixel_0_5_color_column_cls.PNG" width="250">

To get different values for all records pass 

```python
raster = Raster\
            .read\
            .format("shp")\
            .options(
                all_unique="True"
            ).load(shape_path)
raster.show()
```
<img src="https://github.com/Imbruced/sentinel_models/blob/raster_refactor/docs/images/raster_from_shp_pixel_0_5_all_different.PNG" width="250">

