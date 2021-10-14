
"""
Created on Thu Oct  7 10:40:08 2021

@author: anton
"""
import numpy as np
import math as m1
import time
import matplotlib.pyplot as pp



print("\n------Question 1.1 : Matrice A donnée-------\n")     
def DecompositionGS(A):
    """
    Decomposition avec utilisation de l'algorithme de Gram-Schmidt'

    ----------
    A : Matrice A d'entrée'

    Returns
    -------
    Q : Matrice Q orthogonale
    R : Matrice R triangulaire supérieur

    """
    m, n = A.shape
    v = np.zeros((m, n))
    R = np.zeros((n ,n))
    Q = np.zeros((m, n))

    for j in range(0, n):
        v[:, j] = A[:, j]
        for i in range(0,j):
            R[i, j] = np.dot((Q[:,i].conjugate()).T, A[:, j])  #Etape 1 (Rij = < aj , qi >)
            v[:, j] = v[:, j] - R[i, j]*Q[:, i]                #Etape 2 (calcul du vecteur Wj)
            
        R[j, j] = np.linalg.norm(v[:,j], 2)                    #Etape 3 (norme Rjj)
        Q[:, j] = v[:, j]/R[j, j]                              #Etape 4 (Wj/Rjj) 

    return (Q, R)


A = np.array([[12, 24, 46], [58, 5, 88], [2, 55, 32]], dtype='float')
Q, R = DecompositionGS(A)

print ("La matrice A est : \n", A)
print("Le résultat est le suivant :")
print ("\nQ :\n", Q , "\n\nR :\n", R)
DecompositionGS(A)



print("\n------Question 1.2 : Matrice A aléatoire-------\n")
def DecompositionGS(S):
    """
    Résolution QR 
    ----------
    S : Matrice S d'entrée de taille 10*10 à coefficients aléatoires'


    -------
    Q : Matrice Q orthogonale
    R : Matrice R triangulaire supérieure

    """
    m, n = S.shape
    v = np.zeros((m, n))
    R = np.zeros((n, n))
    Q = np.zeros((m, n))

    for j in range(0, n):
        v[:, j] = S[:, j]
        for i in range(0, j):
            R[i, j] = np.dot( (Q[:, i].conjugate()).T, S[:, j] )    #Etape 1 (Rij = < aj , qi >)
            v[:, j] = v[:, j] - R[i, j]*Q[:, i]                     #Etape 2 (calcul du vecteur Wj)
            
        R[j, j] = np.linalg.norm(v[:, j], 2)                        #Etape 3 (norme Rjj)
        Q[:, j] = v[:, j]/R[j, j]                                   #Etape 4 (Wj/Rjj) 

    return (Q, R)


S = np.random.randint(100, size=(10, 10))
Q, R = DecompositionGS(S)

print ("La matrice A est : \n", S)
print("\nLe résulat est le suivant :")
print ("\nQ :\n", Q , "\n\nR :\n", R)
DecompositionGS(S)


print("\n------Exercice 2-------\n")

def ResolutionGS(A, b):
    m, n = A.shape
    X = np.zeros((n, n))
    Y = np.zeros((n, n))
     
    Q, R = DecompositionGS(A)
    
    Y = np.dot(Q.T, b)   
    X = np.dot(np.linalg.inv(R), Y)
    
    return X

A = np.random.rand(12, 12)
b = np.random.rand(12, 1)

x = ResolutionGS(A, b)

print("La résolution de l'équation nous donne x :\n",x)
Vérif = A@x
print("Multiplication de A.x :\n",Vérif)
print("Valeur de b\n",b)


print("\n------Exercice 3-------\n")


def ResolutionSystTriInf(Taug):
    n,m=Taug.shape
    if m !=n+1:
        raise Exception('pas une matrice augmentée')
    x=np.zeros(n)
    for i in range(n):
        somme=0
        for k in range(i):
            somme=somme+x[k]*Taug[i,k]
        x[i]=(Taug[i,-1]-somme)/Taug[i,i]
    return x

def ResolutionSystTriSup(Taug):
    n,m=Taug.shape
    x =np.zeros(n)
    for k in range(n-1,-1,-1):
        S=0
        for j in range(k+1,n):
            S=S+Taug[k,j]*x[j]
        x[k]=(Taug[k,-1]-S)/Taug[k,k]
    return x


def DecompositionLU(A):
    gik_val = []
    n, m = np.shape(A)
    for k in range(0, n-1):
        for i in range(k+1, n):
            gik = A[i, k] / A[k, k]
            gik_val.append(gik)
            A[i, :] = A[i, :] - gik*A[k, :]
    U = A
    print("Upper : ", "\n",  U, "\n")
    L = np.zeros((n, n))
    n, m = np.shape(A)
    k = 0
    for i in range(0, n):
        L[i, k] = gik_val[i] / gik_val[i]
        k += 1
    line = 0
    for k in range(0, n-1):
        for i in range(k+1, n):
            L[i, k] = gik_val[line]
            line += 1
    print("Lower :", "\n", L, "\n")
    return L, U


def ResolutionLU(L,U,B):
    Laug = np.column_stack([L,B])
    Y = ResolutionSystTriInf(Laug)
    Uaug = np.column_stack([U,Y])
    X = ResolutionSystTriSup(Uaug)

    return X

def Solve(A,b):
    x = np.linalg.solve(A,b)
    return x

def QR(A):
    x = np.linalg.qr(A)
    return x

def ResolutionQR(A, b):
    m, n = A.shape
    X = np.zeros((n, n))
    Y = np.zeros((n, n))
     
    Q, R = np.linalg.qr(A)
    
    Y = np.dot(Q.T, b)   
    X = np.dot(np.linalg.inv(R), Y)
    
    return X


def ResolutionGS(A, b):
    m, n = A.shape
    X = np.zeros((n, n))
    Y = np.zeros((n, n))
     
    Q, R = DecompositionGS(A)
    
    Y = np.dot(Q.T, b)   
    X = np.dot(np.linalg.inv(R), Y)
    
    return X

def ResolutionQR(A, b):
    m, n = A.shape
    X = np.zeros((n, n))
    Y = np.zeros((n, n))
     
    Q, R = np.linalg.qr(A)
    
    Y = np.dot(Q.T, b)   
    X = np.dot(np.linalg.inv(R), Y)
    
    return X


def courbe1():
    x = []
    y = []
    y0 = []
    y1 = []
    y2 = []
    for i in range(10,  500, 10):
        A = np.random.rand(i, i)
        B = np.random.rand(i, 1)
        time_init0 = time.time()
        [L, U] = DecompositionLU(A)
        ResolutionLU(L, U, B)
        time_end0 = time.time()
        temps0 = time_end0 - time_init0
        
        time_init = time.time()
        Q, R = DecompositionGS(A)
        ResolutionGS(A, B)
        time_end0 = time.time()
        temps = time_end0 - time_init
        
        time_init2 = time.time()
        Solve(A, B)
        time_end2 = time.time()
        temps2 = time_end2 - time_init2
        
        time_init3 = time.time()
        ResolutionQR(A, B)
        time_end3 = time.time()
        temps3 = time_end3 - time_init3
        
        x.append(i)
        y0.append(temps0)
        y.append(temps)
        y1.append(temps2)
        y2.append(temps3)
    pp.plot(x, y0, label='Resolution LU')
    pp.plot(x, y, label='Decomposition QR')
    pp.plot(x, y1, label='numpy.linalg.solve')
    pp.plot(x, y2, label='numpy.linalg.qr')
    pp.title("Temps de calcul en fonction de la dimension")
    pp.xlabel("Dimension")
    pp.ylabel("Temps en sec")
    pp.legend()
    pp.show()


def courbe2():
    x = []
    y1 = []
    y = []
    y2 = []
    y3 = []
    for i in range(10, 500, 50):
        A = np.array(np.random.random(size=(i, i)))
        B = np.array(np.random.random(size=(i, 1)))
        C = np.copy(A)
        
        L, U = DecompositionLU(A)
        b = ResolutionLU(L, U, B)
        erreur1 = np.linalg.norm(C@b - np.ravel(B))
        
        Q, R = DecompositionGS(A)
        a = ResolutionGS(A, B)
        erreur = np.linalg.norm(A@a - np.ravel(B))
        
        a = Solve(A, B)
        erreur2 = np.linalg.norm(A@a - np.ravel(B))
        
        c = ResolutionQR(A, B)
        erreur3 = np.linalg.norm(A@c - np.ravel(B))
        
        x.append(i)
        y1.append(m1.log(erreur1))
        y.append(m1.log(erreur))
        y2.append(m1.log(erreur2))
        y3.append(m1.log(erreur3))
    pp.plot(x, y1, label='LU')
    pp.plot(x, y, label='Decomposition QR')
    pp.plot(x, y2, label='numpy.linalg.solve')
    pp.plot(x, y3, label='numpy.linalg.qr')
    pp.title("log(||Ax -B||) en fonction de la dimension")
    pp.xlabel("Dimension")
    pp.ylabel("Erreur")
    pp.legend()
    pp.show()


courbe2()

