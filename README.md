# Sentinel models
This library was created to create input data for Neural Network based on
Satellite images data. First use was on setninel2 satellite images but it 
will also work for other data sources. Library is in beta version still,
some of the functionality may not be available. 
## Requirements
## Use cases
Example use 
```python
import matplotlib
from gis.geometry import PolygonFrame
from gis.raster import Raster
import matplotlib.pyplot as plt


matplotlib.use('Qt5Agg')

if __name__ == "__main__":
    shape_path = "~tests\\domy.shp"
    polygon_frame = PolygonFrame.from_file(shape_path)
    main_image = Raster.from_file("~tests\\buildings.tif")
    label_raster = Raster.with_adjustment("from_geo", main_image, polygon_frame)

    plt.imshow(label_raster.array[:, :, 0])
    plt.show()

```

![alt text](https://github.com/Imbruced/sentinel_models/blob/master/tests/example_result.PNG)



