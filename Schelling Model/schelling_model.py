#########################################################################
# This is a basic implementation of Schelling Model in Python without	#
# using any classes. Future implementations can be done with classes,	#
# however the main idea here is to explain the phenomenon of homophily	#
# in society with the help of Schelling's Segregration Model			#
#########################################################################

import numpy as np
import matplotlib.pyplot as plt

# Initializes the world with given number of grids
# Returns random positions for the agents
def initialize(grid_size, T_Agents):
	world = np.zeros([grid_size[0], grid_size[1]])
	while True:
		ind = np.random.randint(min(grid_size), size=(2, world.size))
		ind = np.unique(ind, axis=1)
		if (np.size(ind) > 2*T_Agents):
			break
	np.random.shuffle(ind.T)
	ind = np.delete(ind, np.s_[T_Agents:], axis=1)
	return world, ind

# Populates the world with the given agents
def populate(world, N_Agents1, ind):
	p_world = np.copy(world)
	p_world[ ind[:,:N_Agents1][0], ind[:,:N_Agents1][1] ] = 1
	p_world[ ind[:,N_Agents1:][0], ind[:,N_Agents1:][1] ] = -1
	return p_world

# Evaluates the neighbours
def eval_neighbours(p_world, x1, y1):
	k1 = []
	neighbours = np.array([[x1-1,y1],[x1+1,y1],[x1,y1-1],[x1,y1+1],
						   [x1-1,y1-1],[x1-1,y1+1],[x1+1,y1-1],[x1+1,y1+1]]).T
	for k in range(neighbours.shape[1]):
		try:
			# To reject negative indices
			if (neighbours.T[k][0]<0 or neighbours.T[k][1]<0):
				k1.append(k)
			# To reject out of bound index
			p_world[ neighbours.T[k][0], neighbours.T[k][1] ]
		except IndexError:
			k1.append(k)
			pass
	neighbours = np.delete(neighbours, k1, axis=1)
	return neighbours

# Checks if an Agent is satisfied in the given position
def	check_satisfied(p_world, neighbours2, Agent, threshold):
	u = np.count_nonzero(p_world[ neighbours2[0], neighbours2[1] ] == Agent)
	if (u < threshold):
		return 0
	else:
		return 1

# Finds the unsatisfied agents in the world
def unsatisfied(p_world, threshold):
	u_world = np.zeros(p_world.shape)
	for i in range(p_world.shape[0]):
		for j in range(p_world.shape[1]):
			Agent = p_world[i,j]
			if Agent:
				neighbours = eval_neighbours(p_world, i, j)
				if not check_satisfied(p_world, neighbours, Agent, threshold):
					u_world[i, j] = 1
	return u_world

# Search for the nearest satisfying position for the agent
def search(p_world, neighbours1, old_neighbours, x, y, Agent, threshold, itn):
	# Check if a satisfying position is available at distance 1
	t_neighbours1 = neighbours1.shape[1]
	for i in range(t_neighbours1):
		nx = neighbours1[0][i]
		ny = neighbours1[1][i]
		if not p_world[ nx, ny ]:
			e_neigh = eval_neighbours(p_world, nx, ny)
			if check_satisfied(p_world, e_neigh, Agent, threshold):
				return nx, ny

	# Updating the neighbours of neighbours distance incremented by 1
	if (itn == 0):
		old_neighbours = np.copy(neighbours1)
	new_neighbours = np.array([[],[]])
	for i in range(t_neighbours1):
		temp = eval_neighbours(p_world, neighbours1[0][i], neighbours1[1][i])
		t = []
		# To check and eliminate the priorly checked positions
		for j in range(temp.shape[1]):
			if ((temp.T[j]==[x,y]).all(axis=0) or
				any((temp.T[j]==new_neighbours.T).all(axis=1)) or
				any((temp.T[j]==old_neighbours.T).all(axis=1))):
				t.append(j)
		temp = np.delete(temp, t, axis=1)
		new_neighbours = np.append(new_neighbours, temp, axis=1).astype(int)

	# Store all the checked positions
	old_neighbours = np.append(old_neighbours, new_neighbours.astype(int), axis=1)
	# Spread out till the maximum possible position, if none found, stay
	itn = itn + 1
	if (itn > 2*max(p_world.shape)):
		return x, y
	return search(p_world, new_neighbours, old_neighbours, x, y, Agent, threshold, itn)

# Update the current world by more satisfied world
def new_world(p_world, u_world, threshold):
	i1, j1 = [], []
	for i in range(u_world.shape[0]):
		for j in range(u_world.shape[1]):
			Agent = p_world[i,j]
			if Agent:
				neighbours = eval_neighbours(p_world, i, j)
				itn, o_n = 0, 0
				i1,j1 = search(p_world, neighbours, o_n, i, j, Agent, threshold, itn)
				p_world[i,j] = 0
				p_world[i1,j1] = Agent
	return p_world

# Visualization of the data
def visualize(p_world, iterations):
	X_Agents = np.where(p_world == 1)
	O_Agents = np.where(p_world == -1)
	plt.cla()
	ax.scatter(X_Agents[0], X_Agents[1], c='red', s=20, label='X Agents')
	ax.scatter(O_Agents[0], O_Agents[1], c='blue', s=20, label='O Agents')
	ax.legend()
	ax.grid(True)
	plt.title("World\nIterations:{0}".format(iterations))
	plt.axis([-5,50,-5,50])
	plt.show(block=False)
	fig.canvas.draw()

# Main function
if __name__ == "__main__":
	# Set the variables
	grid_size = (50,50)
	N_Agents1 = 800			# No. of Agent-1
	N_Agents2 = 800			# No. of Agent-O
	threshold = 3			# Satisfaction threshold
	error = 1				# Max. no. of unsatisfied Agents acceptable

	# Build the world with Agents
	world, ind = initialize(grid_size, N_Agents1+N_Agents2)
	p_world = populate(world, N_Agents1, ind)

	# Loop until the world becomes more stabilized
	iterations = 0
	fig, ax = plt.subplots()
	print("Number of Iterations\tNumber of unsatisfied Agents")
	while True:
		visualize(p_world,iterations)
		u_world = unsatisfied(p_world, threshold)
		if ((np.count_nonzero(u_world)<error) or (iterations>500)):
			break
		p_world = new_world(p_world, u_world, threshold)
		iterations = iterations + 1
		print("\t{0}\t\t\t\t{1}".format(iterations,np.count_nonzero(u_world)))

	print("Number of X_Agents:{0}".format(np.count_nonzero(p_world==1)))
	print("Number of O_Agents:{0}".format(np.count_nonzero(p_world==-1)))
	print("Number of unsatisfied agents:{0}".format(np.count_nonzero(u_world)))
	print("Number of iterations:{0}".format(iterations))
