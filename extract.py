#!/usr/bin/python

from PIL import Image
import PIL
import sys
import argparse
import collections
import random
import itertools
from tilecommon import *
from arrange import *

#def image_equals(a,b):
#  return PIL.ImageChops.difference(a,b).getbbox() == None



parser = argparse.ArgumentParser(description="arrange tiles smartly into a tileset")

parser.add_argument("map", help = "Input map (such as from vgmaps)")
#parser.add_argument("-n","--neg", help = "Tiles to exclude")
parser.add_argument("out", help = "output tileset (png)")
parser.add_argument("-tw", "--width", type=int, help = "width of grid", default=16)
parser.add_argument("-th", "--height", type = int, help = "height of grid", default=16)
parser.add_argument("-s","--strategy", help = "what strategy to use for arranging",default = "gibbs-swap-flood-block",choices = ["dumb", "greedy", "gibbs", "gibbs-swap", "gibbs-swap-block", "gibbs-swap-flood-block"])
parser.add_argument("-r","--rubric", help = "What style of scoring to use",default = "linear",choices = ["linear","normal","sqrtnormal","sqrnormal"])
parser.add_argument("-t", "--trials", type=int, help = "number of times to repeat method", default=1)
parser.add_argument("--animate", help = "animate gibbs sampling?", action='store_true')
parser.add_argument("--temperature","--temp", type=float, help = "gibbs sampling inverse temperature", default = -1)
parser.add_argument("--iterations", "--iter", "--iters", type=int,  help = "number of gibbs iterations", default = 10000)

args = parser.parse_args()

stitched, tiles, matrix, best = extract_to_tileset(args)
stitched.save(args.out)
