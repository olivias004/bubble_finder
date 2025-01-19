import vtk
import random
import numpy as np

def render_random_void_surface(void_clusters, min_void_size=100, max_points=5000, cube_size=450):
    """
    Renders a random void from the set of void clusters in 3D using VTK (as a continuous surface),
    and ensures that the axes are uniformly scaled to a cube (e.g., 450x450x450).

    Parameters:
    - void_clusters: List of 3D arrays containing voxel coordinates for each void
    - min_void_size: Minimum size threshold for selecting a larger void
    - max_points: Maximum number of points to render (for downsampling large voids)
    - cube_size: The size of the cube to which all axes will be scaled (default 450x450x450)
    """
    # Filter voids by size to ensure we pick a larger void
    large_voids = [v for v in void_clusters if len(v) >= min_void_size]
    
    if not large_voids:
        print("No large voids available to render.")
        return
    
    # Choose a random large void from the filtered list
    random_void = random.choice(large_voids)
    
    # Downsample if the number of points is greater than max_points
    if len(random_void) > max_points:
        random_void = random_void[np.random.choice(random_void.shape[0], max_points, replace=False)]
        print(f"Downsampled void to {max_points} points.")
    else:
        print(f"Rendering void with {len(random_void)} points.")
    
    # Find the maximum extents in each axis
    min_vals = np.min(random_void, axis=0)
    max_vals = np.max(random_void, axis=0)
    
    # Scale the points to fit within a cube of size cube_size
    scaling_factors = cube_size / (max_vals - min_vals)
    normalized_void = (random_void - min_vals) * scaling_factors  # Rescale points
    
    # Create a vtkPoints object and insert points into it
    vtk_points = vtk.vtkPoints()
    for point in normalized_void:
        vtk_points.InsertNextPoint(point[0], point[1], point[2])

    # Create a polydata object to store the points
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)

    # Apply Delaunay triangulation to create a surface mesh
    delaunay = vtk.vtkDelaunay3D()
    delaunay.SetInputData(polydata)
    delaunay.Update()

    # Create a mapper and actor for the surface
    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputConnection(delaunay.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(1, 0, 0)  # Set the surface color to red

    # Create a renderer, render window, and interactor
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    # Add the actor to the scene
    renderer.AddActor(actor)
    renderer.SetBackground(0.1, 0.2, 0.4)  # Set background color

    # Adjust the camera for proper visibility
    def adjust_camera(renderer, bounds):
        # Calculate the center of the object
        center = [(bounds[1] + bounds[0]) / 2, (bounds[3] + bounds[2]) / 2, (bounds[5] + bounds[4]) / 2]
        max_range = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])

        camera = renderer.GetActiveCamera()
        camera.SetFocalPoint(center)
        camera.SetPosition(center[0] + max_range, center[1] + max_range, center[2] + max_range * 1.5)
        camera.SetViewUp(0, 0, 1)
        renderer.ResetCameraClippingRange()

    # Get the bounds of the void and adjust the camera
    bounds = delaunay.GetOutput().GetBounds()
    adjust_camera(renderer, bounds)

    # Ensure equal scaling on all axes
    renderer.ResetCamera()
    render_window.Render()
    renderer.GetActiveCamera().ParallelProjectionOn()

    # Render and interact
    render_window.Render()
    render_window_interactor.Start()


