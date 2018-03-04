import operator
import numpy as np
import random


def importDS(path, FromLine, ToLine):
    f1 = open(path, 'r')
    ds = []
    for i, line in enumerate(f1):
        if i >= FromLine and i <= ToLine:
            str1 = line.replace('\n', '').split(',')
            for l in str1:
                str2 = l.split(' ')
                str3 = []
                for ll in str2:
                    if ll != '' and ll != ' ':
                        str3.append(ll)
                ds.append(str3)
    return ds

def getDistancInput(path, FromLine, ToLine):
    f1 = open(path, 'r')
    ds = []
    DisList = [[None for _ in range(ToLine-FromLine+2)] for _ in range(ToLine-FromLine+2)]
    DictDist = {}
    for i, line in enumerate(f1):
        if i >= FromLine and i <= ToLine:
            str1 = line.replace('\n', '').split(',')
            for j, l in enumerate(str1):
                DisList[30-(i-7)-1][j] = int(l)
                DisList[j][30-(i-7)-1] = int(l)
                DisList[j][j] = 0
                DictDist[str(30-(i-7)).strip() + '|' + str(j+1).strip()] = int(l)
                DictDist[str(j+1).strip() + '|' + str(30-(i-7)).strip()] = int(l)
                DictDist[str(30-(i-7)).strip() + '|' + str(30-(i-7)).strip()] = 0
    DictDist['1|1'] = 0
    DisList[28][28]=0
    return DictDist,DisList
def roulette_selection(weights):
    sorted_indexed_weights = sorted(enumerate(weights), key=operator.itemgetter(1));
    indices, sorted_weights = zip(*sorted_indexed_weights);
    tot_sum = sum(sorted_weights)
    prob = [x / tot_sum for x in sorted_weights]
    cum_prob = np.cumsum(prob)
    random_num = random.random()
    for index_value, cum_prob_value in zip(indices, cum_prob):
        if random_num < cum_prob_value:
            return index_value

ds1=importDS('bayg29.tsp',37,65)
destdict ,destlist= getDistancInput('bayg29.tsp',8,35)

def ACO_MetaHeuristic(roh,alfa,beta,tedadMorche,DistanceDict,DistanceMatrix,iter):
    ncity = len(destlist)
    tau = (1 - np.diag(np.ones(ncity)))
    delta_tau = (1 - np.diag(np.ones(ncity)))

    oldcurrent = {}
    current = {}
    visited = {}

    for i in range(ncity):
        current[i] = int(np.random.choice(ncity, 1, replace=False))
        current[i]=i
        visited[i] = [current[i]]

    for i1 in range(iter):
        for i2 in range(m):
            p = [];

            # mohasebeye zigma
            zigma = 0
            for k in range(ncity):
                # print('i = '+str(i)+'    k = '+str(k) + '     current[i]='+str(current[i]))
                # print(DistanceMatrix[current[i]][k])
                if (DistanceMatrix[current[i]][k]) > 0:
                    zigma += (tau[current[i]][k] ** alfa) * (DistanceMatrix[current[i]][k] ** beta)
                elif (DistanceMatrix[k][current[i]]) > 0:
                    zigma += (tau[k][current[i]] ** alfa) * (DistanceMatrix[k][current[i]] ** beta)
                else:
                    zigma += 0

            for j in range(ncity):
                if destlist[current[i]][j] > 0:
                    eta = 100.0 / float(DistanceMatrix[current[i]][j])
                elif DistanceMatrix[j][current[i]] > 0:
                    eta = 100.0 / float(DistanceMatrix[j][current[i]])
                else:
                    eta = 0.0

                if j in visited[i]:
                    p.append(0)
                else:
                    if zigma == 0: print('err : destlist[' + current[str(i)] + '][' + str(j) + ']' + str(
                        DistanceMatrix[current[i]][k]) + '    tau[' + str(i) + '][' + str(j) + '] = ' + str(
                        tau[i][j]) + '      current[' + str(i) + ']=' + str(current[i]))
                    if j == 534 and current[i] == 534:
                        p.append(0)
                    else:
                        p.append((tau[i][j] ** alfa) * (eta ** beta) / zigma)


m =50  #tedad morcheh
Q=100
alpha = 2        #Zarib tavani Tao
Beta =1           #Zarib tavani eta
Roh = 0.02         #Darsad Tabkhir
iter = 10           #iteration for all ants




ACO_MetaHeuristic(Roh,alpha,Beta,m,destdict,destlist,iter)
