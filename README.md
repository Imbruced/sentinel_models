# Sentinel models
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

