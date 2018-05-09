import sys

sys.path.insert(0, '.')

from arrange import *
import matplotlib.pyplot as plt
import numpy as np

if len(sys.argv) == 0:
    print("Needs a path to an image")
else:
    path = sys.argv[1]

img_map = Image.open(path).convert("RGB")
tiles, matrix, counts, rubric = extract(img_map,16,16)
rubric_alter = rubric_normal
rubric = rubric_alter(rubric,counts)

ITER = 20000

w=math.ceil(math.sqrt(len(tiles)))
h=len(tiles)//w+1

dataGreedy = []
dataGS     = []
dataGSB    = []
dataGSBF   = []

for i in range(10):
    print("strategy -- greedy:")
    matrixGreedy = arrange_greedy(tiles, rubric, 0, w, h)
    dataGreedy.append([evaluate(matrixGreedy, rubric)] * ITER)

    print("strategy -- gibbs swap:")
    set_gibbs_stats(False, False, 20, ITER)
    matrixGS = arrange_gibbs(tiles, rubric, 0, w, h)
    dataGS.append(get_gibbs_data())

    print("strategy -- gibbs swap block:")
    set_gibbs_stats(True, False, 20,  ITER)
    matrixGSB = arrange_gibbs(tiles, rubric, 0, w, h)
    dataGSB.append(get_gibbs_data())

    print("strategy -- gibbs swap flood block:")
    set_gibbs_stats(True, True, 20, ITER)
    matrixGSBF = arrange_gibbs(tiles, rubric, 0, w, h)
    dataGSBF.append(get_gibbs_data())

plt.plot(    np.mean(dataGS, 0), '.b')
plt.plot(   np.mean(dataGSB, 0), '.r')
plt.plot(  np.mean(dataGSBF, 0), '.g')
plt.plot(np.mean(dataGreedy, 0), '.y')
plt.show()
