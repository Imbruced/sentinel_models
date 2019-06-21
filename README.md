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

```python
shape_path = "/tests/data/shapes/domy.shp"
raster = Raster\
            .read\
            .format("shp")\
            .load()
raster.show()
```
<img src="https://github.com/Imbruced/sentinel_models/blob/raster_refactor/docs/images/raster_from_shp_default_parameters.PNG" width="250">
