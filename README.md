# P2C2: Partial Point Cloud Creator

Load `.stl` model data and create `(.xyz|.stl)` partial point cloud data using openGL depth buffer.

- The point cloud is obtained on the grid in the xy-coordinate direction according to the specification.
    - 構造上xy座標方向においてグリッド上の点群が得られます
- Please check the `create_meta_data` function in `control.py` for detailed parameters.
    - 細かいパラメータの内容は`control.py`の`create_meta_data`関数を確認してください
