import numpy as np

def linear_array(nants, sep=14.6):
    '''Build a linear (east-west) array configuration.
    Arguments:
        nants: integer
            The number of antennas in the configuration.
        sep: float, meters
            The separation between linearly spaced antennas.
    Returns:
        antpos: dictionary
            A dictionary of antenna numbers and positions.  Positions are x,y,z
            in topocentric coordinates, in meters.'''
    antpos = {i: np.array([sep * i, 0, 0]) for i in range(nants)}
    return antpos

def hex_array(hexNum, sep=14.6, split_core=True, outriggers=3):
    '''Build a hexagonal array configuration, nominally matching HERA's
    ideal configuration.
    Arguments:
        hexNum: integer
            The hexagon (radial) number of the core configuration.  Number of 
            core antennas returned is 3N^2 - 3N + 1.
        sep: float, meters
            The separation between hexagonal grid points.
        split_core: bool
            Fractures the hexagonal core into tridrents that subdivide 
            a hexagonal grid.  Loses N antennas, so the number of core antennas
            returned is 3N^2 - 4N + 1.
        outriggers: integer
            Adds R extra rings of outriggers around the core that tile with the
            core to produce a fully-sampled UV plane.  The first ring correponds
            to the exterior of a hexNum=3 hexagon.  Adds 3R^2 + 9R anntenas.
    Returns:
        antpos: dictionary
            A dictionary of antenna numbers and positions.  Positions are x,y,z
            in topocentric coordinates, in meters.'''
    #Main Hex
    positions = []
    for row in range(hexNum - 1, -hexNum + split_core, -1): # the + split_core deletes a row
        for col in range(0, 2 * hexNum - abs(row) - 1):
            xPos = sep * ((-(2 * hexNum - abs(row)) + 2) / 2.0 + col)
            yPos = row * sep * 3**.5 / 2
            positions.append([xPos, yPos, 0])
            
    # unit vectors
    right = sep * np.asarray([1, 0, 0])
    up = sep * np.asarray([0, 1, 0])
    upRight = sep * np.asarray([.5, 3**.5 / 2, 0])
    upLeft = sep * np.asarray([-.5, 3**.5 / 2, 0])
    
    # Split the core into 3 pieces
    if split_core:
        newPos = []
        for i,pos in enumerate(positions):
            theta = np.arctan2(pos[1], pos[0])
            if (pos[0] == 0 and pos[1] == 0):
                newPos.append(pos)
            elif (theta > -np.pi / 3 and theta < np.pi / 3):
                newPos.append(np.asarray(pos) + (upRight + upLeft) / 3)
            elif (theta >= np.pi / 3 and theta < np.pi):
                newPos.append(np.asarray(pos) + upLeft  - (upRight + upLeft) / 3)
            else:
                newPos.append(pos)
        positions = newPos

    # Add outriggers
    if outriggers:
        exteriorHexNum = outriggers + 2
        for row in range(exteriorHexNum - 1, -exteriorHexNum, -1):
            for col in range(2 * exteriorHexNum - abs(row) - 1):
                xPos = ((-(2 * exteriorHexNum - abs(row)) + 2) / 2.0 + col) * sep * (hexNum - 1)
                yPos = row * sep * (hexNum - 1) * 3**.5 / 2
                theta = np.arctan2(yPos, xPos)       
                if ((xPos**2 + yPos**2)**.5 > sep * (hexNum + 1)):
                    # These specific displacements of the outrigger sectors are designed specifically
                    # for redundant calibratability and "complete" uv-coverage, but also to avoid 
                    # specific obstacles on the HERA site (e.g. a road to a MeerKAT atennna).
                    if (theta > 0 and theta <= 2 * np.pi / 3 + .01):
                        positions.append(np.asarray([xPos, yPos, 0]) - 4 * (upRight + upLeft) / 3)
                    elif (theta <= 0 and theta > -2*np.pi/3):
                        positions.append(np.asarray([xPos, yPos, 0]) - 2 * (upRight + upLeft) / 3)
                    else:
                        positions.append(np.asarray([xPos, yPos, 0]) - 3 * (upRight + upLeft) / 3)
                        
    return {i: pos for i,pos in enumerate(np.array(positions))}
