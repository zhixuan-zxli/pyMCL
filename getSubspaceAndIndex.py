import numpy as np
import dolfin

def getSubspaceAndIndex(parentSpace, id):
    subspace, collapsedMap = parentSpace.sub(id).collapse(True)
    a = np.array([[k,v] for (k,v) in collapsedMap.items()])
    subspaceIndex = a[np.argsort(a[:,0]), :]
    assert np.all(subspaceIndex[:-1,0] + 1 == subspaceIndex[1:,0]) # check consecutive
    return (subspace, subspaceIndex)
