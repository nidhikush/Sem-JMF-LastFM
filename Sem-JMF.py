import numpy
import re
import math
import csv
import random
from datetime import date


artist_count = 1000
tag_count = 500
curr_year = int(date.today().year)
user_count = 2100
lbd = 0.75
CNT = 0
CNT_TEST = 0
training_error = 0

def matrix_factorization(R, P, Q, K, steps=5000, gamma=0.003, lambdaa=0.001):
    Q = Q.T
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])
                    for k in range(K):
                        P[i][k] = P[i][k] + gamma * (2 * eij * Q[k][j] - lambdaa * P[i][k])
                        Q[k][j] = Q[k][j] + gamma * (2 * eij * P[i][k] - lambdaa * Q[k][j])
            #print("process " + str(i) + " completed of " + str(len(R)))
        eR = numpy.dot(P,Q)
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
                    for k in range(K):
                        e = e + (lambdaa/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
        #if e < 0.001:
        break
        
    return P, Q.T   


def matrix_factorization_implicit(R, P, Q, K, steps=5000, gamma=0.0002, lambdaa=0.02):
    Q = Q.T
    B = numpy.zeros(shape=(4))
    for step in range(steps):
        for i in range(len(R)):
            B[0] = B[1] = B[2] = B[3] = 0
            for j in range(len(R[i])):               
                if R[i][j] > 3 and R[i][j] <= 4:
                    B[3] = B[3] + 1
                elif R[i][j] > 2 and R[i][j] <= 3:
                    B[2] = B[2] + 1
                elif R[i][j] > 1 and R[i][j] <= 2:
                    B[1] = B[1] + 1
                else:
                    B[0] = B[0] + 1
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    if R[i][j] > 3 and R[i][j] <= 4:
                        div = B[3]
                    elif R[i][j] > 2 and R[i][j] <= 3:
                        div = B[2]
                    elif R[i][j] > 1 and R[i][j] <= 2:
                        div = B[1]
                    else:
                        div = B[0]
                    if div != 0:
                        eij = (R[i][j] - numpy.dot(P[i,:],Q[:,j]))/pow(div, 0.5)
                    else:
                        eij = (R[i][j] - numpy.dot(P[i,:],Q[:,j]))
                    for k in range(K):
                        P[i][k] = P[i][k] + gamma * (2 * eij * Q[k][j] - lambdaa * P[i][k])
                        Q[k][j] = Q[k][j] + gamma * (2 * eij * P[i][k] - lambdaa * Q[k][j])
            print("process " + str(i) + " completed of " + str(len(R)))
        eR = numpy.dot(P,Q)
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
                    for k in range(K):
                        e = e + (lambdaa/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
        if e < 0.001:
            break
    return P, Q.T

def matrix_factorization_implicit_semantic(G, R, S, T, P, Q, C, K, Cat, Tag, c, userMatrix, CNT, gamma = 0.003, steps=300, lambdaa=0.001, alpha = 0.003):
    Q1 = Q
    Q = Q.T
    B = numpy.zeros(shape=(4))
    R = R / 4

    #print (CNT)
    #print (CNT_TEST)

    Q1G = numpy.zeros(shape = (artist_count, tag_count))
    Q1C = numpy.zeros(shape = (artist_count, tag_count))
    catVal = numpy.zeros(shape = (artist_count))
    tagVal = numpy.zeros(shape = (artist_count))
    


    for j in range(len(R[0])):
        for s in range(Tag):
            Q1G[j][s] = numpy.dot(Q1[j,:],G[:,s])
            tagVal[j] = tagVal[j] + c[j][s] * (T[j][s] - Q1G[j][s])
    for j in range(len(R[0])):
        for s in range(Cat):
            Q1C[j][s] = numpy.dot(Q1[j,:],C[:,s])
            catVal[j] = catVal[j] + S[j][s] - Q1C[j][s]    
        
    iterations = 1
    print("Pre calculations completed...")
    for it in range(iterations):
        for step in range(steps):
            #print("userMatrix len = " + str(len(userMatrix)))
            check_cnt = 0
            for i in range(len(userMatrix)):
                B[0] = B[1] = B[2] = B[3] = 0
                for j in userMatrix[i]:
                    if (R[i][j] > 0):
                        if R[i][j] > 0.75 and R[i][j] <= 1:
                            B[3] = B[3] + 1
                        elif R[i][j] > 0.5 and R[i][j] <= 0.75:
                            B[2] = B[2] + 1
                        elif R[i][j] > 0.25 and R[i][j] <= 0.5:
                            B[1] = B[1] + 1
                        else:
                            B[0] = B[0] + 1
                #print ("j = " + str(len(userMatrix[i])))
                for j in userMatrix[i]:
                    if R[i][j] > 0:
                        val = catVal[j]
                        val1 = tagVal[j]
                        if R[i][j] > 0.75 and R[i][j] <= 1:
                           div = B[3]
                        elif R[i][j] > 0.5 and R[i][j] <= 0.75:
                            div = B[2]
                        elif R[i][j] > 0.25 and R[i][j] <= 0.5:
                            div = B[1]
                        else:
                            div = B[0]
                        if div != 0:
                            eij = (R[i][j] - numpy.dot(P[i,:],Q[:,j]))/pow(div, 0.5) + (2 * alpha * val)
                        else:
                            eij = (R[i][j] - numpy.dot(P[i,:],Q[:,j])) + (2 * alpha * val)
                        
                        #for s in range(Cat):
                            #val = val + S[j][s] - Q1C[j][s] 

                        #print("semantic done")
                        #for s in range(Tag):
                            #val1 = val1 + c[j][s] * (T[j][s] - Q1G[j][s])
                        #print("tag done")
                        for k in range(K):
                            P[i][k] = P[i][k] + gamma * (2 * eij * Q[k][j] - lambdaa * P[i][k])
                            #if (P[i][k] < 0):
                                #P[i][k] = 0
                            #if (P[i][k] > 0.5):
                                #P[i][k] = 0.45    
                            Q[k][j] = Q[k][j] + gamma * (2 * eij * P[i][k] - lambdaa * Q[k][j])
                            #if (Q[k][j] < 0):
                                #Q[k][j] = 0
                            #if (Q[k][j] > 0.5):
                                #Q[k][j] = 0.45
                #print("Phase " + str(i) + " Completed outof " + str(len(R)))
            #print(str(step) + " out of " + str(steps))
            eR = numpy.dot(P,Q)
            e = 0
            '''for i in range(len(userMatrix)):
                for j in userMatrix[i]:
                    if R[i][j] > 0:
                        e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
                        #print(R[i][j] - numpy.dot(P[i,:],Q[:,j]))
                        for k in range(K):
                            e = e + (lambdaa/2) * (pow(P[i][k],2) + pow(Q[k][j],2)) / R[i][j]
            if e < 0.001:
                break'''
            for i in range(len(userMatrix)):
                for j in userMatrix[i]:
                    if R[i][j] > 0:
                        e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
                        #print(R[i][j] - numpy.dot(P[i,:],Q[:,j]))
                        for k in range(K):
                            if (R[i][j]  != 0):
                                e = e + (lambdaa/2) * (pow(P[i][k],2) + pow(Q[k][j],2)) / R[i][j]
            e = e / CNT
            e = pow(e, 1/2)
            if e < 0.001:
                break
            training_error = e
            print ("completed " + str(step) + " out of " + str(steps) + " error = " + str(e))
        #lambdaa = lambdaa + 0.008
        print ('Error on training data = '+ str(training_error))
    print("Returning...")
    return P, Q.T

def matrix_factorization_implicit_tag_time(G, R, S, T, P, Q, C, K, Cat, Tag, c, userMatrix, CNT, gamma = 0.003, steps=300, lambdaa=0.001, alpha = 0.003, beta = 0.00025):
    Q1 = Q
    Q = Q.T
    B = numpy.zeros(shape=(4))
    R = R / 4

    #print (CNT)
    #print (CNT_TEST)

    Q1G = numpy.zeros(shape = (artist_count, tag_count))
    Q1C = numpy.zeros(shape = (artist_count, tag_count))
    catVal = numpy.zeros(shape = (artist_count))
    tagVal = numpy.zeros(shape = (artist_count))
    


    for j in range(len(R[0])):
        for s in range(Tag):
            Q1G[j][s] = numpy.dot(Q1[j,:],G[:,s])
            tagVal[j] = tagVal[j] + c[j][s] * (T[j][s] - Q1G[j][s])
            
    for j in range(len(R[0])):
        for s in range(Cat):
            Q1C[j][s] = numpy.dot(Q1[j,:],C[:,s])
            catVal[j] = catVal[j] + S[j][s] - Q1C[j][s]
    
    iterations = 1
    print("Pre calculations completed...")
    for it in range(iterations):
        for step in range(steps):
            if (step % 50 == 0 and step != 0):
                for j in range(len(R[0])):
                    for s in range(Tag):
                        Q1G[j][s] = numpy.dot(Q1[j,:],G[:,s])
                        tagVal[j] = tagVal[j] + c[j][s] * (T[j][s] - Q1G[j][s])
                        
                for j in range(len(R[0])):
                    for s in range(Cat):
                        Q1C[j][s] = numpy.dot(Q1[j,:],C[:,s])
                        catVal[j] = catVal[j] + S[j][s] - Q1C[j][s]
            #print("userMatrix len = " + str(len(userMatrix)))
            check_cnt = 0
            for i in range(len(userMatrix)):
                B[0] = B[1] = B[2] = B[3] = 0
                for j in userMatrix[i]:
                    if (R[i][j] > 0):
                        if R[i][j] > 0.75 and R[i][j] <= 1:
                            B[3] = B[3] + 1
                        elif R[i][j] > 0.5 and R[i][j] <= 0.75:
                            B[2] = B[2] + 1
                        elif R[i][j] > 0.25 and R[i][j] <= 0.5:
                            B[1] = B[1] + 1
                        else:
                            B[0] = B[0] + 1
                #print ("j = " + str(len(userMatrix[i])))
                for j in userMatrix[i]:
                    if R[i][j] > 0:
                        val = catVal[j]
                        val1 = tagVal[j]
                        if R[i][j] > 0.75 and R[i][j] <= 1:
                           div = B[3]
                        elif R[i][j] > 0.5 and R[i][j] <= 0.75:
                            div = B[2]
                        elif R[i][j] > 0.25 and R[i][j] <= 0.5:
                            div = B[1]
                        else:
                            div = B[0]
                        if div != 0:
                            eij = (R[i][j] - numpy.dot(P[i,:],Q[:,j]))/pow(div, 0.5) + (2 * (beta) * val1)
                        else:
                            eij = (R[i][j] - numpy.dot(P[i,:],Q[:,j])) + (2 * (beta) * val1)
                        
                        #for s in range(Cat):
                            #val = val + S[j][s] - Q1C[j][s] 

                        #print("semantic done")
                        #for s in range(Tag):
                            #val1 = val1 + c[j][s] * (T[j][s] - Q1G[j][s])
                        #print("tag done")
                        for k in range(K):
                            P[i][k] = P[i][k] + gamma * (2 * eij * Q[k][j] - lambdaa * P[i][k])
                            #if (P[i][k] < 0):
                                #P[i][k] = 0
                            #if (P[i][k] > 0.5):
                                #P[i][k] = 0.5  
                            Q[k][j] = Q[k][j] + gamma * (2 * eij * P[i][k] - lambdaa * Q[k][j])
                            #if (Q[k][j] < 0):
                                #Q[k][j] = 0
                            #if (Q[k][j] > 0.4):
                                #Q[k][j] = 0.4
                        
                #print("Phase " + str(i) + " Completed outof " + str(len(R)))
            #print(str(step) + " out of " + str(steps))
            eR = numpy.dot(P,Q)
            e = 0
            '''for i in range(len(userMatrix)):
                for j in userMatrix[i]:
                    if R[i][j] > 0:
                        e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
                        #print(R[i][j] - numpy.dot(P[i,:],Q[:,j]))
                        for k in range(K):
                            e = e + (lambdaa/2) * (pow(P[i][k],2) + pow(Q[k][j],2)) / R[i][j]
            if e < 0.001:
                break'''
            for i in range(len(userMatrix)):
                for j in userMatrix[i]:
                    if R[i][j] > 0:
                        e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
                        #print(R[i][j] - numpy.dot(P[i,:],Q[:,j]))
                        for k in range(K):
                            if (R[i][j]  != 0):
                                e = e + (lambdaa/2) * (pow(P[i][k],2) + pow(Q[k][j],2)) / R[i][j]
            e = e / CNT
            e = pow(e, 1/2)
            if e < 0.001:
                break
            training_error = e
            print ("completed " + str(step) + " out of " + str(steps) + " error = " + str(e))
        #lambdaa = lambdaa + 0.008
        print ('Error on training data = '+ str(training_error))
    print("Returning...")
    return P, Q


def matrix_factorization_implicit_semantic_tag_time(G, R, S, T, P, Q, C, K, Cat, Tag, c, userMatrix, CNT, gamma = 0.003, steps=300, lambdaa=0.001, alpha = 0.003):
    Q1 = Q
    Q = Q.T
    B = numpy.zeros(shape=(4))
    R = R / 4

    #print (CNT)
    #print (CNT_TEST)

    Q1G = numpy.zeros(shape = (artist_count, tag_count))
    Q1C = numpy.zeros(shape = (artist_count, tag_count))
    catVal = numpy.zeros(shape = (artist_count))
    tagVal = numpy.zeros(shape = (artist_count))
    


    for j in range(len(R[0])):
        for s in range(Tag):
            Q1G[j][s] = numpy.dot(Q1[j,:],G[:,s])
            tagVal[j] = tagVal[j] + c[j][s] * (T[j][s] - Q1G[j][s])
            
    for j in range(len(R[0])):
        for s in range(Cat):
            Q1C[j][s] = numpy.dot(Q1[j,:],C[:,s])
            catVal[j] = catVal[j] + S[j][s] - Q1C[j][s]
    
    iterations = 1
    print("Pre calculations completed...")
    for it in range(iterations):
        for step in range(steps):
            if (step % 50 == 0 and step != 0):
                for j in range(len(R[0])):
                    for s in range(Tag):
                        Q1G[j][s] = numpy.dot(Q1[j,:],G[:,s])
                        tagVal[j] = tagVal[j] + c[j][s] * (T[j][s] - Q1G[j][s])
                        
                for j in range(len(R[0])):
                    for s in range(Cat):
                        Q1C[j][s] = numpy.dot(Q1[j,:],C[:,s])
                        catVal[j] = catVal[j] + S[j][s] - Q1C[j][s]
            #print("userMatrix len = " + str(len(userMatrix)))
            check_cnt = 0
            for i in range(len(userMatrix)):
                B[0] = B[1] = B[2] = B[3] = 0
                for j in userMatrix[i]:
                    if (R[i][j] > 0):
                        if R[i][j] > 0.75 and R[i][j] <= 1:
                            B[3] = B[3] + 1
                        elif R[i][j] > 0.5 and R[i][j] <= 0.75:
                            B[2] = B[2] + 1
                        elif R[i][j] > 0.25 and R[i][j] <= 0.5:
                            B[1] = B[1] + 1
                        else:
                            B[0] = B[0] + 1
                #print ("j = " + str(len(userMatrix[i])))
                for j in userMatrix[i]:
                    if R[i][j] > 0:
                        val = catVal[j]
                        val1 = tagVal[j]
                        if R[i][j] > 0.75 and R[i][j] <= 1:
                           div = B[3]
                        elif R[i][j] > 0.5 and R[i][j] <= 0.75:
                            div = B[2]
                        elif R[i][j] > 0.25 and R[i][j] <= 0.5:
                            div = B[1]
                        else:
                            div = B[0]
                        if div != 0:
                            eij = (R[i][j] - numpy.dot(P[i,:],Q[:,j]))/pow(div, 0.5) + (2 * alpha * val) + (2 * (beta) * val1)
                        else:
                            eij = (R[i][j] - numpy.dot(P[i,:],Q[:,j])) + (2 * alpha * val) + (2 * (beta) * val1)
                        
                        #for s in range(Cat):
                            #val = val + S[j][s] - Q1C[j][s] 

                        #print("semantic done")
                        #for s in range(Tag):
                            #val1 = val1 + c[j][s] * (T[j][s] - Q1G[j][s])
                        #print("tag done")
                        for k in range(K):
                            P[i][k] = P[i][k] + gamma * (2 * eij * Q[k][j] - lambdaa * P[i][k])
                            #if (P[i][k] < 0):
                                #P[i][k] = 0
                            #if (P[i][k] > 0.5):
                                #P[i][k] = 0.5  
                            Q[k][j] = Q[k][j] + gamma * (2 * eij * P[i][k] - lambdaa * Q[k][j])
                            #if (Q[k][j] < 0):
                                #Q[k][j] = 0
                            #if (Q[k][j] > 0.4):
                                #Q[k][j] = 0.4
                        
                #print("Phase " + str(i) + " Completed outof " + str(len(R)))
            #print(str(step) + " out of " + str(steps))
            eR = numpy.dot(P,Q)
            e = 0
            '''for i in range(len(userMatrix)):
                for j in userMatrix[i]:
                    if R[i][j] > 0:
                        e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
                        #print(R[i][j] - numpy.dot(P[i,:],Q[:,j]))
                        for k in range(K):
                            e = e + (lambdaa/2) * (pow(P[i][k],2) + pow(Q[k][j],2)) / R[i][j]
            if e < 0.001:
                break'''
            for i in range(len(userMatrix)):
                for j in userMatrix[i]:
                    if R[i][j] > 0:
                        e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
                        #print(R[i][j] - numpy.dot(P[i,:],Q[:,j]))
                        for k in range(K):
                            if (R[i][j]  != 0):
                                e = e + (lambdaa/2) * (pow(P[i][k],2) + pow(Q[k][j],2)) / R[i][j]
            e = e / CNT
            e = pow(e, 1/2)
            if e < 0.001:
                break
            training_error = e
            print ("completed " + str(step) + " out of " + str(steps) + " error = " + str(e))
        #lambdaa = lambdaa + 0.008
        print ('Error on training data = '+ str(training_error))
    print("Returning...")
    return P, Q


R = [
     [5,3,0,1],
     [4,0,0,1],
     [1,1,0,5],
     [1,0,0,4],
     [0,1,5,4],
    ]

count = numpy.zeros(shape=(user_count,artist_count))

S = numpy.zeros(shape=(artist_count, 5))
j = 0
i = 0
with open('mappedPCA5.csv') as csvfile:
    readCSV = csv.reader(csvfile,delimiter = ',')
    j = 0
    for row in readCSV:
        for i in range(5):
            S[j][i] = row[i]
        if (j < artist_count - 1):
                j = j + 1    
print("Artist X Categories has been succesfully calculated...")         
            
'''
count = [
        [0,0,15,20],
        [40,10,0,10],
        [10,0,20,15],
        [10,10,0,40],
        [0,0,0,40]
        ]
'''
'''
S = [
    [0, 0.14, 0.22],
    [0, 0.14, 0.22],
    [0, 0.14, 0.22],
    [0, 0.14, 0.22]
    ]'''
'''
T = [
    [10, 20, 20],
    [10, 20, 20],
    [10, 20, 20],
    [10, 20, 20]
    ]
'''

tags = {}

with open("tag1.dat") as f:
    flag = 0;
    for line in f:
        if (flag == 0):
            flag = 1
            continue
        words = line.split()
        tags[int(words[0])] = 1
    #print(tags)
    #print(tags[1])


T = numpy.zeros(shape = (artist_count, tag_count))

with open("user_tag_data.dat") as f:
    flag = 0
    for line in f:
        words = line.split()
        if (flag == 0):
            flag = 1
            continue
        artist_id = int(words[1])
        tag_id = int(words[2])
        #print(artist_id, end=" ")
        #print(tag_id, end=" ")
        #print(time)
        if (tag_id in tags and tag_id < tag_count and artist_id < artist_count):
            tag_id = tag_id
            T[artist_id][tag_id] = T[artist_id][tag_id] + 1


print("Artists X Tags frequency has been succesfully calculated...")

#Hello = numpy.random.rand(10,2)
#Hello = Hello * 4
#print(Hello)
K = 5
Cat = 5
Tag = tag_count
Ta = numpy.zeros(shape=(Tag))
for zz in range(len(T[0])):
    for zz1 in range(len(T)):
        Ta[zz] = Ta[zz] + T[zz1][zz]

#print(Ta)
for zz in range(len(T)):
    for zz1 in range(len(T[0])):
        b = T[zz][zz1] - 1
        a = Ta[zz1]
        #print(a)
        #print(b)
        if (a > 0 and b > 0):
            T[zz][zz1] = round(math.log10((Tag/b))/a, 2)

print("Artists X Tags (T) has been succesfully calculated...")


che = 0
for ri in range(4):
    CNT = 0
    CNT_TEST = 0
    userMatrix = [()] * user_count
    userMatrixTest = [()] * user_count
    with open("user_artists.dat", "r") as f:
        #lines = [line.rstrip('\n') for line in open('user_artists.dat')]
        #print(lines)
        for line in f:
            r = random.randint(0, 4);
            if che == 1:
                final = line.rstrip('\n')
                ff = re.split(r'\t+', final.rstrip('\t'))
                a = int(ff[0])
                b = int(ff[1])
                c = int(ff[2])
                if (a < user_count and b < artist_count and r <= ri):
                    count[a][b] = c
                    userMatrix[a] = userMatrix[a] + (b,)
                    CNT = CNT + 1
                if (a < user_count and b < artist_count and r >= ri + 1):
                    count[a][b] = c
                    userMatrixTest[a] = userMatrixTest[a] + (b,)
                    CNT_TEST = CNT_TEST + 1
            che = 1
    f.close()
    che = 0
    #print("Ratings has been succesfully inputted...")
    count = numpy.array(count)
    freq = numpy.zeros(shape=(user_count, artist_count))
    tr = numpy.zeros(shape=(user_count, artist_count))
    
    #freq = numpy.zeros(shape=(5,4))

    for i in range(len(userMatrix)):
        sum = 0
        for j in userMatrix[i]:
            sum = sum + count[i][j]
        if (sum > 0):
            for j in userMatrix[i]:
                if count[i][j] != 0.0:
                    freq[i][j] = round(count[i][j]/sum, 2)
                    tr[i][j] = freq[i][j]
        

        arr = [None] * len(userMatrix[i])
        arr_idx = 0
        for j in userMatrix[i]:
            arr[arr_idx] = freq[i][j]
            arr_idx = arr_idx + 1
        #arr = sorted(freq[i], reverse=True)
        arr.sort(reverse=True)
        sum = 0
        d = dict()
        if (len(userMatrix[i]) == 0):
            continue
        if arr[0] != 0:
            for j in range(len(arr)):
                if (arr[j] == 0):
                    break
                if j != 0:
                    sum = sum + arr[j-1]
                if j != 0 and arr[j] != arr[j-1]:
                    d[arr[j]] = 4 * (1 - sum)
                if j == 0:
                    d[arr[j]] = 4
            #print (sum)
            for j in userMatrix[i]:
                if (freq[i][j] <= 0):
                    freq[i][j] = 0
                else:
                    freq[i][j] = d[freq[i][j]]
                #print (str(i) + " " + str(j) + " " + str(freq[i][j]))


    for i in range(len(userMatrixTest)):
        sum = 0
        for j in userMatrixTest[i]:
            sum = sum + count[i][j]
        if (sum > 0):
            for j in userMatrixTest[i]:
                if count[i][j] != 0.0:
                    freq[i][j] = round(count[i][j]/sum, 2)
                    tr[i][j] = count[i][j]/sum

        arr = [None] * len(userMatrixTest[i])
        arr_idx = 0
        for j in userMatrixTest[i]:
            arr[arr_idx] = freq[i][j]
            arr_idx = arr_idx + 1
        #arr = sorted(freq[i], reverse=True)
        arr.sort(reverse=True)
        sum = 0
        d = dict()
        if (len(userMatrixTest[i]) == 0):
            continue
        if arr[0] != 0:
            for j in range(len(arr)):
                if (arr[j] == 0):
                    break
                if j != 0:
                    sum = sum + arr[j-1]
                if j != 0 and arr[j] != arr[j-1]:
                    d[arr[j]] = 4 * (1 - sum)
                if j == 0:
                    d[arr[j]] = 3
            #print (sum)
            for j in userMatrixTest[i]:
                if (freq[i][j] <= 0):
                    freq[i][j] = 0
                else:
                    freq[i][j] = d[freq[i][j]]
                #print (str(i) + " " + str(j) + " " + str(freq[i][j]))
        
    freq = numpy.array(freq)
    
    #print("Frequency has been succesfully calculated...")
    R = numpy.array(R)

    '''
    dummyTags = [
        [1, 1, 1, 1],
        [1, 2, 1, 2],
        [1, 3, 2, 0],
        [1, 4, 1, 3],
        [2, 2, 1, 1],
        [2, 3, 3, 1],
        [2, 4, 4, 2],
        [3, 3, 1, 0],
        [3, 4, 4, 1],
        [4, 4, 4, 2]
    ]

    artist = numpy.ndarray((artist_count, ), int)
    tags = numpy.ndarray((tag_count, ), int)

    for i in range(0, artist_count):
        artist [i] = i

    for i in range(0, tag_count):
        tags [i] = i
    '''


    tags = {}

    with open("tag1.dat") as f:
        flag = 0;
        for line in f:
            if (flag == 0):
                flag = 1
                continue
            words = line.split()
            tags[int(words[0])] = 1
        #print(tags)
        #print(tags[1])


    N = len(freq)
    M = len(freq[0])
    C = numpy.random.rand(K, Cat)
    G = numpy.random.rand(K, Tag)
    CXT = numpy.dot(S.T, T)
    C, G = matrix_factorization(CXT, C.T, G.T, K)
    C = C.T
    G = G.T
    P = numpy.random.rand(N,K)
    P = P * 0.5
    #print(P)
    Q = numpy.random.rand(M,K)
    Q = Q * 0.4

    #for cnt in range(100):
    c = numpy.zeros(shape = (artist_count, tag_count))
    postScore = numpy.zeros(shape = (artist_count, tag_count))
    tagSpecificity = numpy.zeros(shape = (artist_count, tag_count))
    time_ratings = freq
    with open("user_tag_data.dat") as f:
        flag = 0
        for line in f:
            words = line.split()
            if (flag == 0):
                flag = 1
                continue
            user_id = int(words[0])
            artist_id = int(words[1])
            tag_id = int(words[2])
            time = curr_year - int(words[5])
            #print(artist_id, end=" ")
            #print(tag_id, end=" ")
            #print(time)
            if (tag_id in tags and tag_id < tag_count and artist_id < artist_count):
                tag_id = tag_id
                time_ratings[i][j] = time_ratings[i][j] * (0.9) ** time
                postScore[artist_id][tag_id] = postScore[artist_id][tag_id] + lbd**time
                tagSpecificity[artist_id][tag_id] = tagSpecificity[artist_id][tag_id] + 1

        '''
        for i in range(len(dummyTags)):
            artist_id = dummyTags[i][1] - 1
            time = dummyTags[i][3]
            tag_id = dummyTags[i][2] - 1
            postScore[artist_id][tag_id] = postScore[artist_id][tag_id] + 0.9**time
            tagSpecificity[artist_id][tag_id] = tagSpecificity[artist_id][tag_id] + 1
        '''

    for  i in range(len(tagSpecificity)):
        for j in range(len(tagSpecificity[i])):
            tagSpecificity[i][j] = math.log10(tagSpecificity[i][j] + 50)
            
    for i in range(artist_count):
        for j in range(tag_count):
            c[i][j] = postScore[i][j] / tagSpecificity[i][j]


    #print("Confidence matrix has been calculated...")

    #N = len(R)
    #M = len(R[0])
    #print(userMatrix)
    #print(Q)
    gamma = 0.001

    #for i in range(100):
    nP, nQ = matrix_factorization_implicit_tag_time(G, freq, S, T, P, Q, C, K, Cat, Tag, c, userMatrix, CNT, gamma)
    #lbd = lbd + 0.01
    #gamma = gamma + 0.0001

    #nP, nQ = matrix_factorization_implicit_semantic_tag(freq, S, T, P, Q, C, K, Cat, Tag)
    #nP, nQ = matrix_factorization_implicit_semantic(freq, S, P, Q, C, K, Cat)
    #nP, nQ = matrix_factorization_implicit(freq, P, Q, K)
    #nP, nQ = matrix_factorization(freq, P, Q, K)
    #nR = numpy.dot(nP, nQ)
    #print (nR)
    #print (nP)
    nQ = nQ.T
    nR = numpy.dot(nP, nQ.T)
    find_max = 0
    for i in range(len(nR)):
        for j in range(len(nR[i])):
            if(nR[i][j] < 0):
                nR[i][j] = 0
            if(nR[i][j] > 1):
                nR[i][j] = 1
            
    #nR = nR / find_max
    error = 0
    for i in range(len(userMatrixTest)):
        for j in userMatrixTest[i]:
            cal_rating = nR[i][j]
            if (cal_rating < 0):
                cal_rating = cal_rating * (-1)
            #print (str(tr[i][j]) + " " + str(cal_rating))
            error = error + pow((tr[i][j]) - cal_rating, 2)
    #print (error)
    error = error / CNT_TEST
    error = pow(error, 1/2)
    print ("Train size = " + str(CNT))
    print ("Test size = " + str(CNT_TEST))
    print ('Error on testing data = '+ str(error))
#print (error)
print("Done...")

