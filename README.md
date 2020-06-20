# Smart Tileset Arranger
*and GMS png-to-room tile converter utility*

## Setup

Before running, install the prerequisites:

```
python -m pip install Pillow
python -m pip install numpy
python -m pip install matplotlib
```

## Tileset Arrangement

This script requires python3 to be installed.

basic usage:

```
python ./extract.py roomImage.png out.png
```

If you want to see it animated, use the `--animate` flag. If you want a quick, naive arrangement (in strips) then set the strategy: `-s dumb`.

## GMS png-to-room utility

If you have a tileset already (one can be procured above):

```
python ./gms_png_to_room.py roomImage.png out.room.gmx --tileset tstImage.png --resource-name tstImage
```

If you don't have a tileset, instead use `--extract tstImageOut.png`:

```
python ./gms_png_to_room.py roomImage.png out.room.gmx --extract tstImageOut.png --resource-name tstImage
```

Note that `--resource-name` refers to the GMS asset name for the tileset.
