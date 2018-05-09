#!usr/bin/python

from PIL import Image
import PIL
import sys
import argparse
import collections
import random
import itertools

def RGBtuple(i):
  return [i & 0xff0000, i & 0x00ff00, i & 0x0000ff]

def extract(img,grid_w,grid_h, tile_list_begin = []):
  tilenums = dict()
  tiles_s = set()
  tiles = []
  
  matrix = [[0 for y in range(img.size[1]//grid_h)] for x in range(img.size[0]//grid_w)]
  
  # the 0 tile
  tilenums[()] = 0
  tiles_s.add(())
  tiles.append(())
  
  for t in tile_list_begin:
    if t not in tiles_s:
      tiles_s.add(t)
      tilenums[t] = len(tiles)
      tiles.append(t)
  
  counts = collections.defaultdict(int)
  
  # blank comparator
  blankimg = PIL.Image.new('RGB', (grid_w, grid_h))
  blanktup = tuple(list(blankimg.getdata()))
  
  # determine tiles in image
  for x in range(0,img.size[0]//grid_w):
    for y in range(0,img.size[1]//grid_h):
      subimg = img.crop((x*grid_w, y*grid_h, (x+1)*grid_w, (y+1)*grid_h))
      tile = tuple(list(subimg.getdata()))
      if tile == blanktup:
        tile = ()
      if not tile in tiles_s:
        tilenums[tile] = len(tiles)
        tiles_s.add(tile)
        tiles.append(tile)
      matrix[x][y] = tilenums[tile]
      counts[tilenums[tile]] += 1
      if len(tiles) > 10000:
        print("exceeded 10,000 unique tiles; aborting")
        sys.exit()
      
  adj = collections.defaultdict(int)
  
  # tabulate adjacencies
  for x, col in enumerate(matrix):
    for y, t in enumerate(col):
      if x > 0:
        adj[(matrix[x-1][y],matrix[x][y],0)] -= 1
      if y > 0:
        adj[(matrix[x][y-1],matrix[x][y],1)] -= 1
  
  return (tiles, matrix, counts, adj)

def stitch(tiles, matrix, width, height):
  img = PIL.Image.new("RGB",(width*len(matrix),height*len(matrix[0])))
  print("Stitching tileset into a " + str(img.size[0]) + "x" + str(img.size[1]) + " image")
  for x in range(len(matrix)):
    for y in range(len(matrix[x])):
      tile = PIL.Image.new("RGB", (width,height))
      data = list(tiles[matrix[x][y]])
      tile.putdata(data)
      img.paste(tile,(x*width,y*height))
  return img
