import sys, random, copy
import numpy as np

# create 4 2D planes to simulate 4x4x4 tic tac toe space for each player
A = [[0 for x in range(4)] for y in range(4)]
B = [[0 for x in range(4)] for y in range(4)]
C = [[0 for x in range(4)] for y in range(4)]
D = [[0 for x in range(4)] for y in range(4)]

A1 = [[0 for x in range(4)] for y in range(4)]
B1 = [[0 for x in range(4)] for y in range(4)]
C1 = [[0 for x in range(4)] for y in range(4)]
D1 = [[0 for x in range(4)] for y in range(4)]

A2 = [[0 for x in range(4)] for y in range(4)]
B2 = [[0 for x in range(4)] for y in range(4)]
C2 = [[0 for x in range(4)] for y in range(4)]
D2 = [[0 for x in range(4)] for y in range(4)]

# create 4 2D planes to keep track of rewards of each tic tac toe position for each player
A1_reward1 = [[0 for x in range(4)] for y in range(4)]
B1_reward1 = [[0 for x in range(4)] for y in range(4)]
C1_reward1 = [[0 for x in range(4)] for y in range(4)]
D1_reward1 = [[0 for x in range(4)] for y in range(4)]

A2_reward2 = [[0 for x in range(4)] for y in range(4)]
B2_reward2 = [[0 for x in range(4)] for y in range(4)]
C2_reward2 = [[0 for x in range(4)] for y in range(4)]
D2_reward2 = [[0 for x in range(4)] for y in range(4)]

# lists to keep track of positions when backtracking for rewards for each player
plane1 = list()
y_coord1 = list()
x_coord1 = list()

plane2 = list()
y_coord2 = list()
x_coord2 = list()


# Object that is evaluated to see if there is a win, A B C D within are the planes with 1 for X and -1 for O
matrices = [A,B,C,D]


# randomly fill board to completion
def fill_randomly():
   count = 0
   player1 = True
   plane_letter = None
   matrixPos = None
   fullMatrix = None
   while count < 64:  # so long as all the spaces have not been occupied
       while True:  # choose a random space that has not yet been occupied in a randomly selected plane
           rand_num = random.randint(1,4)
           if rand_num == 1:
               plane_letter = "A"
               matrixPos = A
           elif rand_num == 2:
               plane_letter = "B"
               matrixPos = B
           elif rand_num == 3:
               plane_letter = "C"
               matrixPos = C
           else:
               plane_letter = "D"
               matrixPos = D
           coordX = random.randrange(4)
           coordY = random.randrange(4)
           if (matrixPos[coordX][coordY]) is 0:
               break
       if player1:  #correspond to either player one or player two's reward matrix
           if plane_letter == "A":
               matrixPos = A1
               fullMatrix = A
           elif plane_letter == "B":
               matrixPos = B1
               fullMatrix = B
           elif plane_letter == "C":
               matrixPos = C1
               fullMatrix = C
           else:
               matrixPos = D1
               fullMatrix = D
           plane1.append(plane_letter)
           x_coord1.append(coordX)
           y_coord1.append(coordY)
           matrixPos[coordX][coordY] = 1
           fullMatrix[coordX][coordY] = 1
       else:
           if plane_letter == "A":
               matrixPos = A2
               fullMatrix = A
           elif plane_letter == "B":
               matrixPos = B2
               fullMatrix = B
           elif plane_letter == "C":
               matrixPos = C2
               fullMatrix = C
           else:
               matrixPos = D2
               fullMatrix = D
           plane2.append(plane_letter)
           x_coord2.append(coordX)
           y_coord2.append(coordY)
           matrixPos[coordX][coordY] = -1
           fullMatrix[coordX][coordY] = -1
       if evaluate():  # return reward depending on if player 1 won
           if player1:
               #print("X (1) wins!")
               reset()
               return 1
           else:
               #print("O (-1) wins!")
               reset()
               return 0
       player1 = not player1
       count += 1
       if count == 64:
           #print("It's a draw!")
           reset()
           return 0.1


# return True if game has been won
def evaluate():
   if horizontal():
       #print("horizontal")
       return True
   if vertical():
       #print("vertical")
       return True
   if diagonal():
       #print("diagonal")
       return True
   if vertical_across():
       #print("vertical across")
       return True
   if horizontal_diagonal():
       #print("horizontal diagonal")
       return True
   if vertical_diagonal():
       #print("vertical diagonal")
       return True
   if diagonal_diagonal():
       #print("diagonal diagonal")
       return True
   return False



# win state for horizontal 4 in a row within plane (16 ways)
def horizontal():
   for matrix in matrices:
       if any(np.sum(matrix, axis=1, dtype=int) == 4):
           return True
       elif any(np.sum(matrix, axis=1, dtype=int) == -4):
           return True
   return False


# win state for vertical 4 in a row within plane (16 ways)
def vertical():
   for matrix in matrices:
       if any(np.sum(matrix, axis=0, dtype=int) == 4):
           return True
       elif any(np.sum(matrix, axis=0, dtype=int) == -4):
           return True
   return False


# win state for diagonal 4 in a row within plane (8 ways)
def diagonal():
   for matrix in matrices:
       if np.sum(np.diagonal(matrix), dtype=int) == 4:
           return True
       elif np.sum(np.diagonal(matrix), dtype=int) == -4:
           return True
       if np.sum(np.diagonal(np.fliplr(matrix)), dtype=int) == 4:
           return True
       elif np.sum(np.diagonal(np.fliplr(matrix)), dtype=int) == -4:
           return True
   return False


# win state for vertical 4 in a row across planes (16 ways)
def vertical_across():
   board = np.stack((A, B, C, D))
   sums = []
   for i in range(4):
       for j in range(4):
           sums.append(np.sum(board[:, i, j], dtype=int))
   if any(np.array(sums) == 4):
       return True
   elif any(np.array(sums) == -4):
       return True
   return False


# win state for diagonal 4 in a row on same row across planes (8 ways)
def horizontal_diagonal():
   board = np.stack((A, B, C, D))
   sums = []
   slice = []
   for j in range(4):
       i = 0
       for k in range(4):
           slice.append(board[i, j, k])
           i += 1
       sums.append(np.sum(slice))
       slice = []

   for j in range(4):
       i = 0
       for k in range(3, -1, -1):
           slice.append(board[i, j, k])
           i += 1
       sums.append(np.sum(slice))
       slice = []

   if any(np.array(sums) == 4):
       return True
   elif any(np.array(sums) == -4):
       return True
   return False


# win state for diagonal 4 in a row on same column across planes (8 ways)
def vertical_diagonal():
   board = np.stack((A, B, C, D))
   sums = []
   slice = []
   for k in range(4):
       i = 0
       for j in range(4):
           slice.append(board[i, j, k])
           i += 1
       sums.append(np.sum(slice))
       slice = []

   for k in range(4):
       i = 0
       for j in range(3, -1, -1):
           slice.append(board[i, j, k])
           i += 1
       sums.append(np.sum(slice))
       slice = []

   if any(np.array(sums) == 4):
       return True
   elif any(np.array(sums) == -4):
       return True
   return False


# win state for diagonal 4 in a row diagonally across planes (4 ways)
def diagonal_diagonal():
   board = np.stack((A, B, C, D))
   sums = []
   slice = []
   # A[0,0] to D[3,3]
   for k in range(4):
       i = k
       j = k
       slice.append(board[i, j, k])
   sums.append(np.sum(slice))
   slice = []

   # A[3,3] to D[0,0]
   i = 0
   for j in range(3, -1, -1):
       k = j
       slice.append(board[i, j, k])
       i += 1
   sums.append(np.sum(slice))
   slice = []

   # A[3,0] to D[0,3]
   i = 0
   for j in range(3, -1, -1):
       k = 3-j
       slice.append(board[i, j, k])
       i += 1
   sums.append(np.sum(slice))
   slice = []

   # A[0,3] to D[3,0]
   i = 0
   for j in range(4):
       k = 3-j
       slice.append(board[i, j, k])
       i += 1
   sums.append(np.sum(slice))

   if any(np.array(sums) == 4):
       return True
   elif any(np.array(sums) == -4):
       return True
   return False


def getPlane(p_current,player):  #function that returns a given player's reward plane corresponding to their position plane
   if player == 1:
       if p_current == "A":
           return A1_reward1
       elif p_current == "B":
           return B1_reward1
       elif p_current == "C":
           return C1_reward1
       else:
           return D1_reward1
   else:
       if p_current == "A":
           return A2_reward2
       elif p_current == "B":
           return B2_reward2
       elif p_current == "C":
           return C2_reward2
       else:
           return D2_reward2


def reset():  # resets all position planes for next trial
   global A,B,C,D,A1,B1,C1,D1,A2,B2,C2,D2, matrices
   A = [[0 for x in range(4)] for y in range(4)]
   B = [[0 for x in range(4)] for y in range(4)]
   C = [[0 for x in range(4)] for y in range(4)]
   D = [[0 for x in range(4)] for y in range(4)]
   matrices = [A,B,C,D]

   A1 = [[0 for x in range(4)] for y in range(4)]
   B1 = [[0 for x in range(4)] for y in range(4)]
   C1 = [[0 for x in range(4)] for y in range(4)]
   D1 = [[0 for x in range(4)] for y in range(4)]

   A2 = [[0 for x in range(4)] for y in range(4)]
   B2 = [[0 for x in range(4)] for y in range(4)]
   C2 = [[0 for x in range(4)] for y in range(4)]
   D2 = [[0 for x in range(4)] for y in range(4)]


def reward1(x):  # reward function that discounts by 0.75 each step forward, directly modifies rewards
   discount = 0.75
   previous_val = 0
   for i in range(len(plane1)):
       current_p = plane1.pop(0)
       current_x = x_coord1.pop(0)
       current_y = y_coord1.pop(0)
       current_plane = getPlane(current_p,1)
       if i == 0:
           current_plane[current_x][current_y] += x
       else:
           current_plane[current_x][current_y] += discount*(previous_val - current_plane[current_x][current_y])
       previous_val = current_plane[current_x][current_y]

       if current_p == "A":
           A1_reward1 = current_plane
       elif current_p == "B":
           B1_reward1 = current_plane
       elif current_p == "C":
           C1_reward1 = current_plane
       else:
           D1_reward1 = current_plane


def reward2(x):  # same as above, but for player 2
   discount = 0.75
   previous_val = 0
   for i in range(len(plane2)):
       current_p = plane2.pop(0)
       current_x = x_coord2.pop(0)
       current_y = y_coord2.pop(0)
       current_plane = getPlane(current_p, 2)
       if i == 0:
           current_plane[current_x][current_y] += x
       else:
           current_plane[current_x][current_y] += discount*(previous_val - current_plane[current_x][current_y])
       previous_val = current_plane[current_x][current_y]

       if current_p == "A":
           A2_reward2 = current_plane
       elif current_p == "B":
           B2_reward2 = current_plane
       elif current_p == "C":
           C2_reward2 = current_plane
       else:
           D2_reward2 = current_plane


def learned_fill():  # similar structure to random_fill, just based on reward function
   count = 0
   player1 = True
   max_plane_reward = None
   opposing_side_plane = None
   plane_letter = None
   A1_r_copy = copy.deepcopy(A1_reward1)  # create copy of rewards since we will be setting maxes to 0 after using to make efficient run time
   B1_r_copy = copy.deepcopy(B1_reward1)
   C1_r_copy = copy.deepcopy(C1_reward1)
   D1_r_copy = copy.deepcopy(D1_reward1)
   A2_r_copy = copy.deepcopy(A2_reward2)
   B2_r_copy = copy.deepcopy(B2_reward2)
   C2_r_copy = copy.deepcopy(C2_reward2)
   D2_r_copy = copy.deepcopy(D2_reward2)

   while count < 64:
       if player1:
           maxA = np.matrix(A1_r_copy).max()  # find max in all 4 planes and play that value
           maxB = np.matrix(B1_r_copy).max()
           maxC = np.matrix(C1_r_copy).max()
           maxD = np.matrix(D1_r_copy).max()
           max_list = [maxA,maxB,maxC,maxD]
           max_val = max(max_list)
           index = max_list.index(max_val)
           if index == 0:  # convert index into plane
               plane_letter = "A"
               max_plane_reward = A1_r_copy
           elif index == 1:
               plane_letter = "B"
               max_plane_reward = B1_r_copy
           elif index == 2:
               plane_letter = "C"
               max_plane_reward = C1_r_copy
           else:
               plane_letter = "D"
               max_plane_reward = D1_r_copy
           max_plane_reward = np.matrix(max_plane_reward)
           max_coords = np.unravel_index(np.argmax(max_plane_reward), max_plane_reward.shape)
           x = max_coords[0]  # get coordinates from the max reward and play in the same spot as max reward
           y = max_coords[1]
           if plane_letter == "A":  # after playing value eliminate from both boards since it has already been used by setting to 0
               A[x][y] = 1
               A1_r_copy[x][y] = 0
               A2_r_copy[x][y] = 0
           elif plane_letter == "B":
               B[x][y] = 1
               B1_r_copy[x][y] = 0
               B2_r_copy[x][y] = 0
           elif plane_letter == "C":
               C[x][y] = 1
               C1_r_copy[x][y] = 0
               C2_r_copy[x][y] = 0
           else:
               D[x][y] = 1
               D1_r_copy[x][y] = 0
               D2_r_copy[x][y] = 0
           plane1.append(plane_letter)
           x_coord1.append(x)
           y_coord1.append(y)
       else:  # same thing for player 2
           maxA = np.matrix(A2_r_copy).max()
           maxB = np.matrix(B2_r_copy).max()
           maxC = np.matrix(C2_r_copy).max()
           maxD = np.matrix(D2_r_copy).max()
           max_list = [maxA, maxB, maxC, maxD]
           max_val = max(max_list)
           index = max_list.index(max_val)
           if index == 0:
               max_plane_reward = A2_r_copy
               plane_letter = "A"
           elif index == 1:
               max_plane_reward = B2_r_copy
               plane_letter = "B"
           elif index == 2:
               max_plane_reward = C2_r_copy
               plane_letter = "C"
           else:
               max_plane_reward = D2_r_copy
               plane_letter = "D"
           max_plane_reward = np.matrix(max_plane_reward)
           max_coords = np.unravel_index(np.argmax(max_plane_reward), max_plane_reward.shape)
           x = max_coords[0]
           y = max_coords[1]
           if plane_letter == "A":  # same as player one, but mark -1 for player 2
               A[x][y] = -1
               A1_r_copy[x][y] = 0
               A2_r_copy[x][y] = 0
           elif plane_letter == "B":
               B[x][y] = -1
               B1_r_copy[x][y] = 0
               B2_r_copy[x][y] = 0
           elif plane_letter == "C":
               C[x][y] = -1
               C1_r_copy[x][y] = 0
               C2_r_copy[x][y] = 0
           else:
               D[x][y] = -1
               D1_r_copy[x][y] = 0
               D2_r_copy[x][y] = 0
           plane2.append(plane_letter)
           x_coord2.append(x)
           y_coord2.append(y)
       if evaluate():
           if player1:
               #print("X (1) wins!")
               reset()
               return 1
           else:
               #print("O (-1) wins!")
               reset()
               return 0
       player1 = not player1
       count += 1
       if count == 64:
           #print("It's a draw!")
           reset()
           return 0.1


def rungame(trials): # method that runs the game
    print("Running... please wait.  If you would like to see trial-by-trial output, uncomment the commented out print statements")
    for i in range(trials):
        if i < trials / 2:  # Exploration for first half
            x = fill_randomly()
            if x == 1:  # if player 1 wins, reward them and clear their lists
                reward1(x)
            elif x == 0: # same with if player 2 wins
                reward2(1)
            else:
                reward1(x)
                reward2(x)
            plane1 = list()
            x_coord1 = list()
            y_coord1 = list()
            plane2 = list()
            x_coord2 = list()
            y_coord2 = list()
        else:  # Exploitation
            y = learned_fill()  # same logic as above
            if y == 1:
                reward1(y)
            elif y == 0:
                reward2(1)
            else:
                reward1(y)
                reward2(y)
            plane1 = list()
            x_coord1 = list()
            y_coord1 = list()
            plane2 = list()
            x_coord2 = list()
            y_coord2 = list()
    print("For " + str(trials) + " trials")  # output results here
    print("Player 1 (X)")
    print(np.matrix(A1_reward1))
    print(np.matrix(B1_reward1))
    print(np.matrix(C1_reward1))
    print(np.matrix(D1_reward1))
    print("Player 2 (O)")
    print(np.matrix(A2_reward2))
    print(np.matrix(B2_reward2))
    print(np.matrix(C2_reward2))
    print(np.matrix(D2_reward2))


# get user inputted number of trials
num1 = 0
num2 = 0
num3 = 0

try:
# get params from command line, show error and terminate program if file not included
   num1 = sys.argv[1]
   num2 = sys.argv[2]
   num3 = sys.argv[3]
   if num1 >= num2 or num2 >= num3:
       print("Parameters must be increasing in value.")
       sys.exit(0)
except IndexError:
   print("File not recognized")
   sys.exit(0)
print(num1, num2, num3)
rungame(int(num1))
rungame(int(num2))
rungame(int(num3))
