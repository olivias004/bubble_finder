import numpy as np

def find_axes(positions):
	"""
	Purpose
    -------


    Calling Sequence
    ----------------
    .. code-block:: python

        from bubble_finder.find_axes import find_axes

        axes = find_axes(positions)


    Input Parameters
    ----------------
    positions:
        
    Output Parameters
    -----------------
    axes:
    

	"""
	axes = []

	for i in range(3):
		values = positions[:,i]
		lower_limit, upper_limit = np.percentile(values, [4,96])
		axis = upper_limit - lower_limit
		axes.append(axis)

	return axes


