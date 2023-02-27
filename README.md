# **P2C2**: Partial Point Cloud Creator

![partial stl image](./image/image_stl.png)

Load `.stl` model data and create `(.xyz|.bin|.stl(ascii)|.stl(binary))` partial point cloud data **using openGL depth buffer**.

- The point cloud is obtained on the grid in the xy-coordinate direction according to the specification.
    - 構造上xy座標方向においてグリッド上の点群が得られます
- Please check the `create_meta_data` function in `control.py` for detailed parameters.
    - 細かいパラメータの内容は`control.py`の`create_meta_data`関数を確認してください

## Essential Packages

Check [python.dockerfile](./python.dockerfile), please.

## RUN

In Docker container...

```
/P2C2$ make run
```
or
```
/P2C2$ xvfb-run python <main.py | other_root.py>
```

- Subcommands of `make` are defined in the [Makefile in the PartialPointCloudCreator directory](./PartialPointCloudCreator/Makefile).
    - [PartialPointCloudCreatorディレクトリのMakefile](./PartialPointCloudCreator/Makefile)に `make` のサブコマンドは定義されています
- `xvfb-run` is a command to run headless.
    - `xvfb-run`はheadless(表面的な画面生成無し)で動作させるためのコマンドです


## Load `.bin` format 

The structure of the bin format is as follows...

```
[4 byte unsigned int] vertex size
(repeat <vertex size> times...):
    [4 byte float] vertex x coordinate
    [4 byte float] vertex y coordinate
    [4 byte float] vertex z coordinate

```

