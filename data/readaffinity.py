import argparse
import matplotlib.pyplot as plot
import matplotlib.cm as cm
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.csgraph as csgraph

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Eigencuts Affinity Matrix', \
        epilog = 'lol pygencuts', add_help = 'How to use', \
        prog = 'python readaffinity.py <Mahout Outfile>')
    parser.add_argument('-i', '--input', required = True,
        help = 'Path to input affinity matrix.')
    parser.add_argument('-d', '--dimensions', required = True, type = int,
        help = 'Number of dimensions in the affinity matrix.')
    parser.add_argument('-t', '--type', choices = ['txt', 'img'], required = True,
        help = 'Type of data that created the affinity matrix.')

    # Optional arguments.
    parser.add_argument('-o', '--original', default = None,
        help = 'If input data type is "txt", this is the path to the original cartesian data.')
    parser.add_argument('--height', type = int, default = -1,
        help = 'If input data type is "img", this is the height of the original image.')
    parser.add_argument('--width', type = int, default = -1,
        help = 'If input data type is "img", this is the width of the original image.')

    args = vars(parser.parse_args())
    A = sparse.lil_matrix((args['dimensions'], args['dimensions']))

    # Read the data.
    for line in file(args['input']):
        # It will be of the format: <key>\t<comma-separated values>
        key, val = line.strip().split("\t")
        i = int(key)

        # Loop through the values.
        row = map(float, val.split(","))
        for j in xrange(0, len(row)):
            if row[j] > 0.0:
                A[i, j] = 1.0
                A[j, i] = 1.0

    # Find the connected components!
    numConn, connMap = csgraph.connected_components(A, directed = False)

    # Print them out.
    print 'Found %s clusters.' % numConn
    if args['type'] == 'img':
        plot.figure(0)    
        plot.imshow(np.reshape(connMap, newshape = (args['height'], args['width'])),
            interpolation = 'nearest')
        plot.show()
    else:
        data = np.loadtxt(args['original'], delimiter = ",")
        plot.figure(0)
        colormap = cm.get_cmap("jet", numConn)
        colorvals = colormap(np.arange(numConn))
        colors = [colorvals[connMap[i]] for i in range(0, np.size(connMap))]
        for i in range(0, data.shape[0]):
            plot.plot(data[i, 0], data[i, 1], marker = 'o', c = colors[i])
        plot.show()
