from tilecommon import *
from PIL import Image
import PIL
import sys
import argparse
import collections
import random
import math
import itertools
import time
import copy
import numpy
import numpy.random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib

def rubric_linear(adj,counts):
  return adj

def rubric_normal(adj,counts):
  for key in adj:
    a = key[0]
    b = key[1]
    orientation = key[2]
    adj[key] /= max(counts[a],counts[b])
  adj[0, 0, 0] = -0.03
  adj[0, 0, 1] = -0.03
  return adj

def rubric_sqrnormal(adj,counts):
  for key in adj:
    a = key[0]
    b = key[1]
    orientation = key[2]
    adj[key] /= max(counts[a],counts[b])**2
  return adj

def rubric_sqrtnormal(adj,counts):
  for key in adj:
    a = key[0]
    b = key[1]
    orientation = key[2]
    adj[key] /= (max(counts[a],counts[b]))**0.5
  adj[0, 0, 0] = -0.6
  adj[0, 0, 1] = -0.6
  return adj

def arrange_dumb(tiles,rubric,score,width,height):
  matrix = [[0 for y in range(height)] for x in range(width)]
  for t in range(len(tiles)):
     matrix[t % width][t // width] = t
  return matrix

def retrieve(matrix,x,y,default = 0):
  w = len(matrix)
  h = len(matrix[0])
  if x>=0 and x < w and y >= 0 and y < h:
    return matrix[x][y]
  return default

def arrange_greedy(tiles,rubric,score,width,height):
  matrix = [[0 for y in range(height)] for x in range(width)]
  x = width//2
  y = height//2
  unused = set(range(1,len(tiles)))
  frontierset = set()
  frontierset.add((x,y))
  while (len(frontierset) > 0 and len(unused) > 0):
    pop = random.sample(frontierset,1)[0]
    frontierset.remove(pop)
    x = pop[0]
    y = pop[1]
    if x < 0 or x >= len(matrix) or y < 0 or y >= len(matrix[0]):
      continue
    if matrix[pop[0]][pop[1]] != 0:
      continue


    l = retrieve(matrix,x-1,y)
    r = retrieve(matrix,x+1,y)
    u = retrieve(matrix,x,y-1)
    d = retrieve(matrix,x,y+1)

    to_add = []
    if retrieve(matrix,x-1,y,-1) == 0:
      to_add.append((x-1,y))

    best = random.sample(unused,1)[0]
    bestscore = 0
    if l != 0 or r != 0 or u != 0 or d != 0:
      unused_iter = list(unused)
      random.shuffle(unused_iter)
      for t in unused_iter:
        sc = rubric[(l,t,0)] + rubric[(t,r,0)] + rubric[(u,t,1)] + rubric[(t,d,1)]
        if sc < bestscore:
          best = t
          bestscore = sc

    unused.remove(best)
    matrix[x][y] = best

    #print("" + str(x) + "," + str(y) + ": " + str(best) + "(s" + str(bestscore) + ")")

    Rp = [(-1,-1),(-1,0),(-1,1),
          ( 0,-1),       ( 0,1),
          ( 1,-1),( 1,0),( 1,1)]
    random.shuffle(Rp)
    for (i,j) in Rp:
      frontierset.add((i+x,j+y))

  return matrix

# calculates the relative score about a particular
# element
def evaluateBlanket(ar, rubric, x, y):
    sc = 0
    if x > 0:
        sc += rubric[(ar[x-1][y]),(ar[x][y]),0]
    if y > 0:
        sc += rubric[(ar[x][y-1]),(ar[x][y]),1]
    if x < len(ar) - 1:
        sc += rubric[(ar[x][y]),(ar[x+1][y]),0]
    if y < len(ar[x]) - 1:
        sc += rubric[(ar[x][y]),(ar[x][y + 1]),1]
    return sc

GIBBS_TEMPERATURE = 3.5

# swaps-resamples the given tile
def swap_resample_single(matrix, rubric, score, width, height):
    global GIBBS_TEMPERATURE
    xt = random.randint(0, width - 1)
    yt = random.randint(0, height - 1)
    defScore =  evaluateBlanket(matrix, rubric, xt, yt)
    scoreCoords = []
    probs = []
    scoreShift = []
    for xj in range(0, width):
        for yj in range(0, height):
            prevScore = evaluateBlanket(matrix, rubric, xj, yj)

            # swap
            tmp = matrix[xt][yt]
            matrix[xt][yt] = matrix[xj][yj]
            matrix[xj][yj] = tmp

            newScore = evaluateBlanket(matrix, rubric, xj, yj) + evaluateBlanket(matrix, rubric, xt, yt)
            diffScore = newScore - defScore - prevScore
            scoreShift.append(diffScore)
            scoreCoords.append((xj, yj))
            probs.append(math.exp(min(40, max(-40, -diffScore * GIBBS_TEMPERATURE))))

            # swap
            tmp = matrix[xt][yt]
            matrix[xt][yt] = matrix[xj][yj]
            matrix[xj][yj] = tmp

    #numpy.testing.assert_array_equal(pmat, matrix)

    # sample
    sump = sum(probs)
    probvec = [sp/sump for sp in probs]
    coordIndex = numpy.random.choice(range(len(probs)), 1, p=probvec)[0]
    coord = scoreCoords[coordIndex]
    # swap
    if (xt, yt) != coord:
        tmp = matrix[xt][yt]
        matrix[xt][yt] = matrix[coord[0]][coord[1]]
        matrix[coord[0]][coord[1]] = tmp
    # return score difference
    return scoreShift[coordIndex]

# samples from the r-ratio geometric distribtion constrained at most n.
# 1 <= r <= n
def sample_geometric(r, n):
    # normalizing constant z
    rn = math.pow(r, n)
    z = (1 - rn)/(1 - r)
    p = 1/z
    u = random.random();
    for i in range(n - 1):
        if u <= p:
            return i + 1
        else:
            u -= p
            p *= r
    return n

# evaluates the given rectangle's perimeter (for gibbs sampling)
def evaluateRectanglePerimeter(ar, rubric, x1, y1, x2, y2):
    #return evaluateRectangle(ar, rubric, x1, y1, x2, y2)
    sc = 0
    sc += evaluateLine(ar, rubric, x1, y1, x1, y2)
    sc += evaluateLine(ar, rubric, x1, y1, x2, y1)
    sc += evaluateLine(ar, rubric, x2, y1, x2, y2)
    sc += evaluateLine(ar, rubric, x1, y2, x2, y2)
    return sc

# evaluates the given boundary
# needs (x1, y1), (x2, y2) to be orthogonal
def evaluateLine(ar, rubric, x1, y1, x2, y2):
    sc = 0
    if x1 == x2: # vertical
        if x1 > 0 and x1 < len(ar):
            for y in range(y1, y2):
                sc += rubric[(ar[x1-1][y]),(ar[x1][y]),0]
    elif y1 == y2: # horizontal
        if y1 > 0 and y1 < len(ar[0]):
            for x in range(x1, x2):
                sc += rubric[(ar[x][y1-1]),(ar[x][y1]),1]
    else:
        assert(False)
    return sc

# evaluates the full contents of the given area, including boundaries
def evaluateRectangle(ar, rubric, x1, y1, x2, y2):
  sc = 0
  for x in range(x1, x2+1):
      sc += evaluateLine(ar, rubric, x, y1, x, y2)
  for y in range(y1, y2+1):
      sc += evaluateLine(ar, rubric, x1, y, x2, y)
  return sc

# swaps contents of two rectangles
# reverse: swap order is reversed (important if rectangles overlap
# because the permutation is potentially of order greater than 2)
def rectangleSwap(matrix, x1, x2, y1, y2, width, height, reverse = False):
    xRange = range(width) if not reverse else range(width - 1, -1, -1)
    yRange = range(height) if not reverse else range(height - 1, -1, -1)
    #print(x1 + width, y1 + height, reverse)
    for x in xRange:
        for y in yRange:
            tmp = matrix[x + x1][y + y1]
            matrix[x + x1][y + y1] = matrix[x + x2][y + y2]
            matrix[x + x2][y + y2] = tmp

# finds an semi-optimal block in the matrix
def floodBlock(matrix, rubric, threshold):
    width = len(matrix)
    height = len(matrix[0])
    x1 = numpy.random.randint(width)
    y1 = numpy.random.randint(height)
    x2 = x1 + 1
    y2 = y1 + 1
    for i in range(sample_geometric(0.9,max(len(matrix),len(matrix[x1])) - 2) + 2):
        NO_GROW = 0
        UP_GROW = 1
        DOWN_GROW = 2
        LEFT_GROW = 3
        RIGHT_GROW = 4
        score = [threshold,
                 evaluateLine(matrix, rubric, x1, y1, x2, y1)/float(x2 - x1),
                 evaluateLine(matrix, rubric, x1, y2, x2, y2)/float(x2 - x1),
                 evaluateLine(matrix, rubric, x1, y1, x1, y2)/float(y2 - y1),
                 evaluateLine(matrix, rubric, x2, y1, x2, y2)/float(y2 - y1)]
        BEST = np.argmin(score)
        if score[BEST] >= threshold:
            BEST = NO_GROW
        if BEST == NO_GROW:
            break
        if BEST == UP_GROW:
            if y1 == 0:
                break;
            y1 -= 1
        if BEST == DOWN_GROW:
            if y2 == height:
                break;
            y2 += 1
        if BEST == LEFT_GROW:
            if x1 == 0:
                break
            x1 -= 1
        if BEST == RIGHT_GROW:
            if x2 == width:
                break
            x2 += 1
    return x1, y1, x2 - x1, y2 - y1

def swap_resample_block(matrix, rubric, score, width, height):
    global GIBBS_TEMPERATURE
    global GIBBS_FLOOD
    SAMPLE_P = 0.9
    didFlood = False
    if not GIBBS_FLOOD or random.random() < 0.4:
        blockWidth = sample_geometric(SAMPLE_P, width - 1)
        blockHeight = sample_geometric(SAMPLE_P, height - 1)
        x1 = numpy.random.randint(width - blockWidth)
        assert (x1 <= width - blockWidth)
        y1 = numpy.random.randint(height - blockHeight)
        assert (y1 <= height - blockHeight)
    else:
        didFlood = True
        threshold = - random.random() * GIBBS_TEMPERATURE
        x1, y1, blockWidth, blockHeight = floodBlock(matrix, rubric, threshold)

    if blockWidth == 1 and blockHeight == 1:
        return swap_resample_single(matrix, rubric, score, width, height), 1, 1, False, didFlood

    fullInitScore = evaluate(matrix, rubric) # used to compare when debugging
    defScore = evaluateRectanglePerimeter(matrix, rubric, x1, y1, x1 + blockWidth, y1 + blockHeight)

    scoreCoords = []
    probs = []
    scoreShift = []
    scoreOverlaps = []
    prevScore = 0
    newScore = 0

    # print(x1, y1, x1+blockWidth, y1+blockHeight,width,height)

    # generate sampling probabilities:
    for x2 in range(width - blockWidth + 1):
        for y2 in range(height - blockHeight + 1):
            #determine if rectangles overlap
            hOverlaps = True
            vOverlaps = True
            if x1 > x2 + blockWidth:
                hOverlaps = False
            if x2 > x1 + blockWidth:
                hOverlaps = False
            if y1 > y2 + blockHeight:
                vOverlaps = False
            if y2 > y1 + blockHeight:
                vOverlaps = False
            overlaps = hOverlaps and vOverlaps
            minX = min(x1, x2)
            minY = min(y1, y2)
            maxX = max(x1, x2) + blockWidth
            maxY = max(y1, y2) + blockHeight

            # evaluate initial pre-swap score
            if overlaps:
                prevScore = evaluateRectangle(matrix, rubric, minX, minY, maxX, maxY)
            else:
                prevScore = evaluateRectanglePerimeter(matrix, rubric, x2, y2, x2 + blockWidth, y2 + blockHeight) + defScore

            # swap
            rectangleSwap(matrix, x1, x2, y1, y2, blockWidth, blockHeight)

            # calculate score difference
            if overlaps:
                newScore = evaluateRectangle(matrix, rubric, minX, minY, maxX, maxY)
            else:
                newScore =   evaluateRectanglePerimeter(matrix, rubric, x1, y1, x1 + blockWidth, y1 + blockHeight) \
                           + evaluateRectanglePerimeter(matrix, rubric, x2, y2, x2 + blockWidth, y2 + blockHeight)

            # unswap
            rectangleSwap(matrix, x1, x2, y1, y2, blockWidth, blockHeight, True)

            diffScore = newScore - prevScore

            scoreCoords.append((x2, y2))
            scoreShift.append(diffScore)
            probs.append(math.exp(min(40, max(-40, -diffScore * GIBBS_TEMPERATURE))))
            scoreOverlaps.append(overlaps)

    # sample
    sump = sum(probs)
    probvec = [sp/sump for sp in probs]
    coordIndex = numpy.random.choice(range(len(probs)), 1, p=probvec)[0]
    coord = scoreCoords[coordIndex]

    # perform operation
    rectangleSwap(matrix, x1, coord[0], y1, coord[1], blockWidth, blockHeight)

    assert((evaluate(matrix, rubric) - fullInitScore - scoreShift[coordIndex]) < 0.000001)

    return scoreShift[coordIndex], blockWidth, blockHeight, scoreOverlaps[coordIndex], didFlood

GIBBS_BLOCK = False
GIBBS_FLOOD = False
GIBBS_ANIMATE = False
GIBBS_ITERATIONS = 1000

def set_gibbs_stats(bloc, flood, tmp, iterations):
    global GIBBS_BLOCK
    global GIBBS_FLOOD
    global GIBBS_ITERATIONS
    global GIBBS_TEMPERATURE
    GIBBS_TEMPERATURE = tmp
    GIBBS_BLOCK = bloc
    GIBBS_FLOOD = flood
    GIBBS_ITERATIONS = iterations

gibbs_data = []

def get_gibbs_data():
    global gibbs_data
    return gibbs_data

def arrange_gibbs(tiles, rubric, score, width, height):
    global GIBBS_BLOCK
    global GIBBS_ANIMATE
    global GIBBS_ITERATIONS
    global gibbs_data
    gibbs_data = []
    print(GIBBS_BLOCK)
    matrix = arrange_greedy(tiles, rubric, score, width, height)
    initial = evaluate(matrix, rubric)
    cumulativeScore = initial
    bestMatrix = copy.deepcopy(matrix)
    bestScore = initial
    bestScoreTime = 0

    MODE_WARMUP = 1
    MODE_PRIMED = 2
    MODE_SAVEBEST = 3

    mode = MODE_WARMUP

    maxTime = GIBBS_ITERATIONS
    for i in range(maxTime):
        gibbs_data.append(cumulativeScore)
        scoreDiff = 0
        if mode == MODE_WARMUP or not GIBBS_BLOCK or random.random() < 0.2:
            scoreDiff = swap_resample_single(matrix, rubric, score, width, height)
        else:
            scoreDiff, bWidth, bHeight, overlaps, didFlood = swap_resample_block(matrix, rubric, score, width, height)

        cumulativeScore += scoreDiff

        if cumulativeScore < bestScore:
            # cache best
            diff = cumulativeScore - bestScore
            bestScore = cumulativeScore
            bestScoreTime = i
            if mode > MODE_WARMUP:
                bestMatrix = copy.deepcopy(matrix)
                print("improved:", bestScore, "(", diff ,")")
        elif i - bestScoreTime > 150 and mode == MODE_WARMUP:
            # switch mode
            mode = MODE_PRIMED
            bestScoreTime = i
            print("Mode <- 2")
        elif i - bestScoreTime > 300:
            bestScoreTime = i
            mode = MODE_SAVEBEST

        # display
        if i % 100 == 0:
            currentScore = evaluate(matrix, rubric)
            cumulativeScore = currentScore # floating point correction
            print("score:",currentScore,"; iteration", i,"/",maxTime)
            if GIBBS_ANIMATE:
                showImage(stitch(tiles, matrix, 16, 16))

    if mode == MODE_SAVEBEST:
        matrix = bestMatrix

    print("initial score: ")
    print(initial)
    print("vs improved: ")
    print(evaluate(matrix, rubric))
    print("calculated: ", bestScore)
    return matrix

def evaluate(ar, rubric):
  return evaluateRectangle(ar, rubric, 0, 0, len(ar), len(ar[0]))

fig = -1

def showImage(image, block = False):
    global fig
    if fig == -1:
        plt.ion()
        fig = plt.figure()
    else:
        plt.clf()
    try:
        plt.imshow(np.asarray(image))
        fig.canvas.draw()
        time.sleep(1e-6)
        plt.show(block)
    except:
        fig = plt.figure()

def extract_to_tileset(args):
  global GIBBS_BLOCK
  global GIBBS_ANIMATE
  global GIBBS_FLOOD
  global GIBBS_TEMPERATURE
  global GIBBS_ITERATIONS
  print("reading " + args.map)
  img_map = Image.open(args.map).convert("RGB")

  print ("extracting tiles")
  tiles, matrix, counts, rubric = extract(img_map,args.width,args.height)

  print("computing " + args.rubric + " rubric")

  rubric_alter = rubric_linear
  if args.rubric == "linear":
    rubric_alter = rubric_linear
  if args.rubric == "normal":
    rubric_alter = rubric_normal
  if args.rubric == "sqrtnormal":
    rubric_alter = rubric_sqrtnormal
  if args.rubric == "sqrnormal":
    rubric_alter = rubric_sqrnormal

  rubric = rubric_alter(rubric,counts)

  print(str(len(tiles)) + " unique tiles found")

  print ("arranging tileset using " + args.strategy + " strategy")

  arrange = arrange_dumb
  if args.strategy == "dumb":
    arrange = arrange_dumb
  if args.strategy == "greedy":
    arrange = arrange_greedy
  if args.strategy == "gibbs" or args.strategy == "gibbs-swap":
    arrange = arrange_gibbs
    GIBBS_BLOCK = False
  if args.strategy == "gibbs-swap-block":
    arrange = arrange_gibbs
    GIBBS_BLOCK = True
  if args.strategy == "gibbs-swap-flood-block":
      arrange = arrange_gibbs
      GIBBS_FLOOD = True
      GIBBS_BLOCK = True
  if args.animate:
      GIBBS_ANIMATE = True
  if args.temperature > 0:
      GIBBS_TEMPERATURE = args.temperature
  GIBBS_ITERATIONS = args.iterations

  w=math.ceil(math.sqrt(len(tiles)))
  h=len(tiles)//w+1

  w += 1
  h += 0

  if args.trials > 1:
      print("Attempt 1")
  best = arrange(tiles,rubric,-30,w,h)
  bestscore = evaluate(best,rubric)
  print("Score: " + str(bestscore))
  for i in range(args.trials - 1):
      print("Attempt " + str(i + 2))
      ar = arrange(tiles,rubric,-30,w,h)
      sc = evaluate(ar,rubric)
      if sc < bestscore:
          best = ar
          bestscore = sc
          print("Score: " + str(bestscore))

  print("Final score: " + str(bestscore))

  stitched = stitch(tiles, best, args.width, args.height)
  showImage(stitched, True)
  return (stitched, tiles, matrix, best)
