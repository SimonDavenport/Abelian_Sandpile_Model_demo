# A toy implementation of the abelian Sandpile model

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as la
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg

class Sandpile:

    def __init__(self, adjacency, sinks):
        self._dense_limit = 10
        self.adjacency = adjacency
        self.sinks = sinks
        self.reduced_laplacian = None
        self.reduced_laplacian = self.get_reduced_laplacian()

    def get_reduced_laplacian(self):
        if self.reduced_laplacian is None:
            outdegs = np.array(self.adjacency.sum(axis=0)).flatten()
            mask = np.ones(self.adjacency.shape[0], dtype=bool)
            mask[self.sinks] = False
            print('computing laplacian')
            if sparse.isspmatrix_csr(self.adjacency):
                laplacian = sparse.diags([outdegs], [0], format='csr') - self.adjacency
            else:
                laplacian = np.diag(outdegs) - self.adjacency
            print('DONE')
            print('removing sinks to get reduced laplacian')
            reduced_laplacian = sparse.csr_matrix(laplacian[mask, :][:, mask])
            print('DONE')
            return reduced_laplacian
        else:
            return self.reduced_laplacian

    def get_group_order(self):
        ''' |S(G)| = det(L)'''
        if self.get_reduced_laplacian().shape[0] < self._dense_limit:
            return np.round(np.linalg.det(self.get_reduced_laplacian().todense()))
        else:
            return 0

    def get_max_configuration(self):
        '''The maximal stable configuration on G is c_max = sum_v (outdeg(v)-1) v'''
        print('Computing max configuration')
        outdegs = np.array(self.adjacency.sum(axis=0)).flatten()
        return np.delete(outdegs-1, sinks, 0)

    def fire_script(self, configuration, script):
        ''' c' = c - L s'''
        new_configuration = configuration - self.get_reduced_laplacian().dot(script)
        return new_configuration, np.any(new_configuration<0)

    def generate_firing_script(self, configuration_start, configuration_target):
        '''sigma = L^-1 (c - c') iff c = c' mod L '''
        print('Generating firing script')
        x = la.bicgstab(self.get_reduced_laplacian(), configuration_start - configuration_target, tol=1e-2)
        if x[1]==0:
            print('Sucessfully found a script')
            return x[0]
        else:
            return np.zeros(length(configuration_start))

    def get_stabilizer_script(self, configuration):
        '''Generate an idea script that assumes all intervening steps are valid, then perform a binary search 
        in the assumed intervening steps in order to find the first invalid move.'''

        ideal_script = np.floor(self.generate_firing_script(configuration, np.zeros(len(configuration))))

        #ideal_distance = sum(ideal_script ** 2)

        #current_distance = ideal_distanceyo, cehcjking in again

        #while(self.fire_script(configuration, np.array([9, 11])))

        test = self.fire_script(configuration, ideal_script)

        return test

    def stabilize(self, configuration):
        '''Run a 1 direction at a time stabilization, checing for further valid firing directions 
        at each iteration until no further valid firings are found'''
        print('Stabilizing given configuration')
        current_configuration = np.copy(configuration)
        max_configuration = self.get_max_configuration()
        cumulative_script = np.zeros(len(configuration))
        iter_count = 0
        while(np.any(current_configuration > max_configuration)):
            #script = current_configuration > max_configuration
            script = np.floor(current_configuration / (max_configuration + 1))
            cumulative_script += script
            current_configuration = self.fire_script(current_configuration, script)[0]
            iter_count += 1
            if iter_count % 100 == 0 :
                print('At iteration ' + str(iter_count))
        print('Stabilized in ' + str(iter_count) + ' iterations ; max stabilize script ' + str(np.max(cumulative_script)))
        return current_configuration, cumulative_script

        #new_configuration = self.fire_script(configuration, self.get_stabilizer_script(configuration))
        #return new_configuration

    def compute_identity(self):
        '''I = (2 * c_max - (2 * c_max)^O )^O'''
        print('Computing identity')
        return self.stabilize(2 * self.get_max_configuration() - self.stabilize(2 * self.get_max_configuration())[0])[0]

    def compute_null_configuration(self):
        '''c_null = c_max + [1, ..., 1] - (c_max + [1, ..., 1])^O'''
        print('Computing null configuration')
        max_configuration = self.get_max_configuration()
        return max_configuration + 1 - self.stabilize(max_configuration + 1)[0]

    def test_recurrence(self, configuration):
        ''' If c is recurrent then (c_null + c)^O = c
        '''
        return np.all(self.stabilize(self.compute_null_configuration() + configuration)[0] == configuration)

    #def test_recurrence(self, configuration, recurrent_configuration):
    #    ''' A configuration c is recurrent if and only if there exists a configuration b>=0 such that
    #        c = c(c_max + b)^O
    #    '''
    #    return np.all(configuration==self.stabilize(recurrent_configuration + self.get_max_configuration))

class AnimateSquareSandpile:
    def __init__(self, configuration, sandpile, dim, fig):
        self.dim = dim
        self.current_configuration = configuration
        self.cumulative_script = np.zeros(len(configuration))
        self.max_configuration = sandpile.get_max_configuration()
        self.sandpile = sandpile
        self.plot_canvas = fig.gca().imshow(self.current_configuration.reshape((self.dim, self.dim)), 
                                        vmin=0, vmax=6, animated=True)
        #self.plot_canvas = fig.gca().imshow(self.cumulative_script.reshape((self.dim, self.dim)), 
        #                                    vmin=0, vmax=712, animated=True)

    def update(self):
        if np.any(self.current_configuration > self.max_configuration):
            script = np.floor(self.current_configuration / (self.max_configuration + 1))
            self.cumulative_script += script
            self.current_configuration = self.sandpile.fire_script(self.current_configuration, script)[0]
            self.plot_canvas.set_data(self.current_configuration.reshape((self.dim, self.dim)))
            #self.plot_canvas.set_data(self.cumulative_script.reshape((self.dim, self.dim)))


if __name__ == '__main__':

    '''
    sinks = [2]
    adjacency = sparse.csr_matrix(([3, 5, 2, 1, 2, 1], ([0, 1, 1, 2, 2, 2], [1, 0, 2, 0, 1, 2])), shape=(3, 3))

    test = Sandpile(adjacency, sinks)

    starting_configuration = np.array([4, 5])

    next_configuration = test.fire_script(starting_configuration, [1, 2])[0]
    print(next_configuration)

    next_configuration = test.stabilize(starting_configuration)
    print(next_configuration)

    identity = test.compute_identity()

    print(identity)

    script = test.generate_firing_script(np.array([4, 5]), np.array([4, 0]))

    print(script)
    
    next_configuration = test.stabilize(np.array([20, 20]))
    print(next_configuration)

    script = test.get_stabilizer_script(np.array([20, 20]))

    print(script)
    '''
    # for diamond lattice
    '''
    sinks = [3]
    adjacency = np.array([[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 1]])

    test = Sandpile(adjacency, sinks)

    identity = test.compute_identity()
    '''
    # try for a square lattice

    fig = plt.figure()

    dim = 152
    dim1 = 152

    sinks = [i + dim*j for i in range(0, dim) for j in range(0, dim1) if (i==0 or i==(dim-1) or j==0 or j==(dim1-1))]

    if dim==dim1:
        toe = sparse.diags([1, 1], [1, -1], shape=(dim, dim), format='csr')
        adjacency = sparse.kron(toe, sparse.eye(dim)) + sparse.kron(sparse.eye(dim), toe)

        #plt.imshow(adjacency.todense())
        #plt.show()

        square_sandpile = Sandpile(adjacency, sinks)

        result = square_sandpile.stabilize(2*square_sandpile.get_max_configuration())[0]

        #result2 = square_sandpile.stabilize(2*square_sandpile.get_max_configuration()-result)

        animation_helper = AnimateSquareSandpile(2*square_sandpile.get_max_configuration(), square_sandpile, dim-2, fig)

        def animate(frame):
            animation_helper.update()
            return animation_helper.plot_canvas

        ani = animation.FuncAnimation(fig, animate, interval=5, blit=False, repeat=False, frames=10000)
        ani.save(r'C:\Users\Simon\Work\VisualStudioProjects\Abelian_Sandpile_Model_demo\animations\output' + str(dim-2) + '.mp4')


        #test_configuration = np.zeros((dim-2)*(dim1-2))
        #test_configuration = 2 * square_sandpile.get_max_configuration()

        #result = square_sandpile.stabilize(test_configuration)

        #plt.imshow(result.reshape([dim-2, dim1-2]))

        #plt.show()

        #identity = square_sandpile.compute_identity()

        #plt.imshow(identity.reshape([dim-2, dim1-2]))

        #plt.show()

        #x = square_sandpile.test_recurrence(identity)

        #x1 = np.all(square_sandpile.stabilize(identity + square_sandpile.get_max_configuration()) == square_sandpile.get_max_configuration())

        #real_identity = np.array([1, 3, 3, 1, 3, 2, 2, 3, 3, 2, 2, 3, 1, 3, 3, 1])

        #y = square_sandpile.test_recurrence(real_identity)

        #y1 = np.all(square_sandpile.stabilize(real_identity + square_sandpile.get_max_configuration()) == square_sandpile.get_max_configuration())

