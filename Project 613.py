import random
import time
import matplotlib.pyplot as plt
import copy
from copy import deepcopy

#Asks user for number of rows and columns
R = int(input("How many rows in the matrix? "))




#Decides if entries are random or not
Q = str(input("Will the entries be random? "))
#If random, choose random float (0,1), times by 10, then round to 8 places
if (Q.lower()=="yes" or Q.lower()=="y" or Q.lower()=="yeah"):
    # Sets up matrices A,b as empty lists
    amatrix = []
    bmatrix = []
    #print("Matrix A:")
    for i in range(R):
        #Sets up partial matrix to separate rows and columns
        pmatrix=[]
        for j in range(R):
            #Appends random values to the final matrix
            pmatrix.append(10*random.random())
        #Prints the row just completed to output something that looks like a matrix to the user
        #print(pmatrix)
        #Adds everything in each row as a list to the matrix
        amatrix.append(pmatrix)
    #Creates a copy of matrix A that will remain unchanged throughout to produce difference vector below
    perma = copy.deepcopy(amatrix)

    for i in range(R):
        b = 10*random.random()
        bmatrix.append(b)
    #print(" ")
    #print(" ")
    #print("Matrix b:")
    #print(bmatrix)
    #print(" ")
    #print(" ")
    permb = copy.deepcopy(bmatrix)

    print("Augmented Matrix")
    for i in range(R):
        print("[")
        for j in range(R):
            print(amatrix[i][j], end=" ")
        print(" | " + str(bmatrix[i]) + "]")
    print(" ")
    print(" ")


    start = time.time()
    #Gaussian Elimination  (Makes A into an upper triangular matrix)
    for i in range(R-1):
        d=1/amatrix[i][i]
        for j in range(i+1,R):
            #Stores current value of lower triangle elements
            aji = []
            aji.append(amatrix[j][i])

            #Gives new value of a_ji
            amatrix[j][i] = amatrix[j][i]*d

            # Adjusts matrix b accordingly
            bmatrix[j] = bmatrix[j] - bmatrix[i] * amatrix[j][i]

            #Makes nondiagonal elements 0
            amatrix[j][i] = amatrix[j][i]*amatrix[i][i]-amatrix[i][i]*amatrix[j][i]
            for k in range(i+1,R):
                amatrix[j][k] = amatrix[j][k]-aji[0]*d*amatrix[i][k]


    print("New Matrix After Gaussian Elimination:")
     #Prints new matrix A
    for i in range(R):
        print("[")
        for j in range(R):
            print(amatrix[i][j], end=" ")
        print(" | " + str(bmatrix[i]) + "]")
    print(" ")
    #print("New b")
    #print(bmatrix)
    #print(" ")
    solns = []
    #Back substitution
    #Finds first solution by dividing the last element of b and A
    xn = bmatrix[R-1]/amatrix[R-1][R-1]
    solns.append(xn)
    for i in range(R-1,-1,-1):
        xi = bmatrix[i]
        for j in range(i+1,R):
            xi = xi-amatrix[i][j]*solns[R-j-1]
        #Checks if the value of xi after iterating is equivalent to one already in solution set
        #If it isn't, it gets added, otherwise it's skipped
        if xi/amatrix[i][i] not in solns:
            xi = xi/amatrix[i][i]
            solns.append(xi)
      #Outputs the solutions found with back substitution
    print("Solutions:")
    solns.reverse()
    print(solns)
    print("")
    print("")

    #Finds product Ax
    prod = []
    for i in range(R):
        tot = 0
        for j in range(R):
            product = perma[i][j]*solns[j]
            tot += product
        prod.append(tot)

    diffvec = []
    for i in range(R):
        diff = prod[i]-permb[i]
        diffvec.append(diff)
    print("Difference vector between solutions and b" )
    print(diffvec)
    print(" ")
    print(" ")

    onorm = 0
    for i in range(R):
        onorm += abs(diffvec[i])
    print("1-norm:")
    print(onorm)

    tnorm = 0
    for i in range(R):
        tnorm += abs(diffvec[i])**2
    tnorm = tnorm**0.5
    print("2-norm:")
    print(tnorm)

    inorm = 0
    for i in range(R):
        if abs(diffvec[i]) > inorm:
            inorm = abs(diffvec[i])
    print(str(float('inf')) + "-norm")
    print(inorm)
    print(" ")
    print(" ")

    print(str(time.time()-start) + " seconds to process")

    #Takes the new Matrix A and creates its transpose
    tam = []
    for i in range(R):
        ptam = []
        for j in range(R):
            ptamij = amatrix[j][i]
            ptam.append(ptamij)
        #print(ptam)
        tam.append(ptam)
    print("")
    print("")
    print("")

    #Adds the transpose of Matrix A with itself and multiplies result by half
    jmat = []
    print("Symmetric Matrix")
    for i in range(R):
        pjmat = []
        for j in range(R):
            jmatij = 0.5*(amatrix[i][j] + tam[i][j])
            pjmat.append(jmatij)
        print(pjmat)
        jmat.append(pjmat)

    #jmat is now a symmetric matrix
    #Jacobi Method
    #Calculates the outer norm of A
    out = 0
    for i in range(R):
        for j in range(i+1,R):
            na = (jmat[i][j])**2
            out += na
    #print(out)

    ep = 10**-10

    outs = []
    outs.append(out)

    while out > ep:
        #Initializes max value
        #Sets them up as -1 to avoid confusion
        #i.e. so the base case isn't that jmat[0][0] is the max
        max = -1
        ival = -1
        jval = -1
        #Checks each value in jmat to find the largest element
        for i in range(R):
            #Only need range i+1-R because the matrix is symmetric
            #Also ensures we focus on non-diagonal elements
            for j in range(i+1,R):
                if abs(jmat[i][j]) > max:
                    max = abs(jmat[i][j])
                    ival = i
                    jval = j

        if max != 0:
            theta = (jmat[ival][ival]-jmat[jval][jval])/(2*jmat[ival][jval])

            t= 0

            if theta != 0:
                sign = theta/abs(theta)
                t = sign*(abs(theta)+(1+theta**2)**0.5)*-1

            c = (1+t**2)**-0.5

            s = c*t
            #print(theta,t,c,s)

            bi = [0]*R
            bj = [0]*R

            # Full Jacobi rotation for the diagonal entries
            bi[ival] = s**2*jmat[jval][jval]+c**2*jmat[ival][ival]+2*s*c*jmat[jval][ival]
            bj[jval] = c**2*jmat[jval][jval]+s**2*jmat[ival][ival]-2*s*c*jmat[jval][ival]

            for l in range(R):
                if (l != ival and l != jval):
                    bi[l] = c*jmat[ival][l]+s*jmat[jval][l]
                    bj[l] = -s*jmat[ival][l]+c*jmat[jval][l]

            out -= jmat[ival][jval]*jmat[ival][jval]
            outs.append(out)
            for l in range(R):
                jmat[ival][l] = bi[l]
                jmat[l][ival] = bi[l]
                jmat[jval][l] = bj[l]
                jmat[l][jval] = bj[l]
    print('')
    print("")

    print("Eigenvalues:")
    for i in range(R):
        print(jmat[i][i])

    print("")
    print("")
    #Outputs the amount of time it's been since the user chose a random matrix system
    print(str(time.time()-start) + " seconds to process")

    #Creates a list of index places in above list out
    ind = []
    for i in range(len(outs)):
        dex = i+1
        ind.append(dex)

    #Plots the points (index number,outer norm)
    plt.plot(ind,outs,"ro")
    plt.xlabel("Iteration")
    plt.ylabel("Outer Norm")
    plt.show()





else:
    amatrix = []
    bmatrix = []
    for i in range (R):
        pam = []
        for j in range(R):
            A = float(input("Type the elements of matrix A row wise: "))
            pam.append(A)
        amatrix.append(pam)
    for j in range (R):
        b = float(input("Type the elements of matrix b row wise: "))
        bmatrix.append(b)
    print("")
    print("")
    print("Augmented Matrix")
    for i in range(R):
        print("[")
        for j in range(R):
            print(amatrix[i][j], end=" ")
        print(" | " + str(bmatrix[i]) + "]")
    print(" ")
    print(" ")

    perma = copy.deepcopy(amatrix)
    permb = copy.deepcopy(bmatrix)

    start = time.time()
    # Gaussian Elimination  (Makes A into an upper triangular matrix)
    for i in range(R - 1):
        d = 1 / amatrix[i][i]
        for j in range(i + 1, R):
            # Stores current value of lower triangle elements
            aji = []
            aji.append(amatrix[j][i])

            # Gives new value of a_ji
            amatrix[j][i] = amatrix[j][i] * d

            # Adjusts matrix b accordingly
            bmatrix[j] = bmatrix[j] - bmatrix[i] * amatrix[j][i]

            # Makes nondiagonal elements 0
            amatrix[j][i] = amatrix[j][i] * amatrix[i][i] - amatrix[i][i] * amatrix[j][i]
            for k in range(i + 1, R):
                amatrix[j][k] = amatrix[j][k] - aji[0] * d * amatrix[i][k]

    print("New Matrix After Gaussian Elimination:")
    # Prints new matrix A
    for i in range(R):
        print("[")
        for j in range(R):
            print(amatrix[i][j], end=" ")
        print(" | " + str(bmatrix[i]) + "]")
    print(" ")
    # print("New b")
    # print(bmatrix)
    # print(" ")
    solns = []
    # Back substitution
    # Finds first solution by dividing the last element of b and A
    xn = bmatrix[R - 1] / amatrix[R - 1][R - 1]
    solns.append(xn)
    for i in range(R - 1, -1, -1):
        xi = bmatrix[i]
        for j in range(i + 1, R):
            xi = xi - amatrix[i][j] * solns[R - j - 1]
        # Checks if the value of xi after iterating is equivalent to one already in solution set
        # If it isn't, it gets added, otherwise it's skipped
        if xi / amatrix[i][i] not in solns:
            xi = xi / amatrix[i][i]
            solns.append(xi)
    # Outputs the solutions found with back substitution
    print("Solutions:")
    solns.reverse()
    print(solns)
    print("")
    print("")

    # Finds product Ax
    prod = []
    for i in range(R):
        tot = 0
        for j in range(R):
            product = perma[i][j] * solns[j]
            tot += product
        prod.append(tot)

    diffvec = []
    for i in range(R):
        diff = prod[i] - permb[i]
        diffvec.append(diff)
    print("Difference vector between solutions and b")
    print(diffvec)
    print(" ")
    print(" ")

    onorm = 0
    for i in range(R):
        onorm += abs(diffvec[i])
    print("1-norm:")
    print(onorm)

    tnorm = 0
    for i in range(R):
        tnorm += abs(diffvec[i]) ** 2
    tnorm = tnorm ** 0.5
    print("2-norm:")
    print(tnorm)

    inorm = 0
    for i in range(R):
        if abs(diffvec[i]) > inorm:
            inorm = abs(diffvec[i])
    print(str(float('inf')) + "-norm")
    print(inorm)
    print(" ")
    print(" ")

    print(str(time.time() - start) + " seconds to process")

    # Takes the new Matrix A and creates its transpose
    tam = []
    for i in range(R):
        ptam = []
        for j in range(R):
            ptamij = amatrix[j][i]
            ptam.append(ptamij)
        # print(ptam)
        tam.append(ptam)
    print("")
    print("")
    print("")

    # Adds the transpose of Matrix A with itself and multiplies result by half
    jmat = []
    print("Symmetric Matrix")
    for i in range(R):
        pjmat = []
        for j in range(R):
            jmatij = 0.5 * (amatrix[i][j] + tam[i][j])
            pjmat.append(jmatij)
        print(pjmat)
        jmat.append(pjmat)

    # jmat is now a symmetric matrix
    # Jacobi Method
    # Calculates the outer norm of A
    out = 0
    for i in range(R):
        for j in range(i + 1, R):
            na = (jmat[i][j]) ** 2
            out += na
    # print(out)

    ep = 10 ** -10

    outs = []
    outs.append(out)

    while out > ep:
        # Initializes max value
        # Sets them up as -1 to avoid confusion
        # i.e. so the base case isn't that jmat[0][0] is the max
        max = -1
        ival = -1
        jval = -1
        # Checks each value in jmat to find the largest element
        for i in range(R):
            # Only need range i+1-R because the matrix is symmetric
            # Also ensures we focus on non-diagonal elements
            for j in range(i + 1, R):
                if abs(jmat[i][j]) > max:
                    max = abs(jmat[i][j])
                    ival = i
                    jval = j

        if max != 0:
            theta = (jmat[ival][ival] - jmat[jval][jval]) / (2 * jmat[ival][jval])

            t = 0

            if theta != 0:
                sign = theta / abs(theta)
                t = sign * (abs(theta) + (1 + theta ** 2) ** 0.5) * -1

            c = (1 + t ** 2) ** -0.5

            s = c * t
            # print(theta,t,c,s)

            bi = [0] * R
            bj = [0] * R

            # Full Jacobi rotation for the diagonal entries
            bi[ival] = s ** 2 * jmat[jval][jval] + c ** 2 * jmat[ival][ival] + 2 * s * c * jmat[jval][ival]
            bj[jval] = c ** 2 * jmat[jval][jval] + s ** 2 * jmat[ival][ival] - 2 * s * c * jmat[jval][ival]

            for l in range(R):
                if (l != ival and l != jval):
                    bi[l] = c * jmat[ival][l] + s * jmat[jval][l]
                    bj[l] = -s * jmat[ival][l] + c * jmat[jval][l]

            out -= jmat[ival][jval] * jmat[ival][jval]
            outs.append(out)
            for l in range(R):
                jmat[ival][l] = bi[l]
                jmat[l][ival] = bi[l]
                jmat[jval][l] = bj[l]
                jmat[l][jval] = bj[l]
    print('')
    print("")

    print("Eigenvalues:")
    for i in range(R):
        print(jmat[i][i])

    print("")
    print("")
    # Outputs the amount of time it's been since the user chose a random matrix system
    print(str(time.time() - start) + " seconds to process")

    # Creates a list of index places in above list out
    ind = []
    for i in range(len(outs)):
        dex = i + 1
        ind.append(dex)

    # Plots the points (index number,outer norm)
    plt.plot(ind, outs, "ro")
    plt.xlabel("Iteration")
    plt.ylabel("Outer Norm")
    plt.show()

