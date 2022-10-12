import vtk
import numpy as np


def create_camera_polydata(R, t, only_polys=False):
    """Creates a vtkPolyData object with a camera mesh: https://github.com/lmb-freiburg/demon"""
    import vtk
    cam_points = np.array([
        [0, 0, 0],
        [-1, -1, 1.5],
        [1, -1, 1.5],
        [1, 1, 1.5],
        [-1, 1, 1.5],
        [-0.5, 1, 1.5],
        [0.5, 1, 1.5],
        [0, 1.2, 1.5],
        [1, -0.5, 1.5],
        [1, 0.5, 1.5],
        [1.2, 0, 1.5]]
    )
    cam_points = (0.05 * cam_points - t).dot(R)

    vpoints = vtk.vtkPoints()
    vpoints.SetNumberOfPoints(cam_points.shape[0])
    for i in range(cam_points.shape[0]):
        vpoints.SetPoint(i, cam_points[i])
    vpoly = vtk.vtkPolyData()
    vpoly.SetPoints(vpoints)

    poly_cells = vtk.vtkCellArray()

    if not only_polys:
        line_cells = vtk.vtkCellArray()

        line_cells.InsertNextCell(5)
        line_cells.InsertCellPoint(1)
        line_cells.InsertCellPoint(2)
        line_cells.InsertCellPoint(3)
        line_cells.InsertCellPoint(4)
        line_cells.InsertCellPoint(1)

        line_cells.InsertNextCell(3)
        line_cells.InsertCellPoint(1)
        line_cells.InsertCellPoint(0)
        line_cells.InsertCellPoint(2)

        line_cells.InsertNextCell(3)
        line_cells.InsertCellPoint(3)
        line_cells.InsertCellPoint(0)
        line_cells.InsertCellPoint(4)

        # x-axis indicator
        line_cells.InsertNextCell(3)
        line_cells.InsertCellPoint(8)
        line_cells.InsertCellPoint(10)
        line_cells.InsertCellPoint(9)
        vpoly.SetLines(line_cells)
    else:
        # left
        poly_cells.InsertNextCell(3)
        poly_cells.InsertCellPoint(0)
        poly_cells.InsertCellPoint(1)
        poly_cells.InsertCellPoint(4)

        # right
        poly_cells.InsertNextCell(3)
        poly_cells.InsertCellPoint(0)
        poly_cells.InsertCellPoint(3)
        poly_cells.InsertCellPoint(2)

        # top
        poly_cells.InsertNextCell(3)
        poly_cells.InsertCellPoint(0)
        poly_cells.InsertCellPoint(4)
        poly_cells.InsertCellPoint(3)

        # bottom
        poly_cells.InsertNextCell(3)
        poly_cells.InsertCellPoint(0)
        poly_cells.InsertCellPoint(2)
        poly_cells.InsertCellPoint(1)

        # x-axis indicator
        poly_cells.InsertNextCell(3)
        poly_cells.InsertCellPoint(8)
        poly_cells.InsertCellPoint(10)
        poly_cells.InsertCellPoint(9)

    # up vector (y-axis)
    poly_cells.InsertNextCell(3)
    poly_cells.InsertCellPoint(5)
    poly_cells.InsertCellPoint(6)
    poly_cells.InsertCellPoint(7)

    vpoly.SetPolys(poly_cells)

    return vpoly

def create_camera_actor(R, t):
    """https://github.com/lmb-freiburg/demon
    Creates a vtkActor object with a camera mesh
    """
    vpoly = create_camera_polydata(R, t)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(vpoly)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().LightingOff()
    actor.GetProperty().SetLineWidth(2)

    return actor

def create_pointcloud_polydata(points, colors=None):
    """https://github.com/lmb-freiburg/demon
    Creates a vtkPolyData object with the point cloud from numpy arrays

    points: numpy.ndarray
        pointcloud with shape (n,3)

    colors: numpy.ndarray
        uint8 array with colors for each point. shape is (n,3)

    Returns vtkPolyData object
    """
    vpoints = vtk.vtkPoints()
    vpoints.SetNumberOfPoints(points.shape[0])
    for i in range(points.shape[0]):
        vpoints.SetPoint(i, points[i])
    vpoly = vtk.vtkPolyData()
    vpoly.SetPoints(vpoints)

    if not colors is None:
        vcolors = vtk.vtkUnsignedCharArray()
        vcolors.SetNumberOfComponents(3)
        vcolors.SetName("Colors")
        vcolors.SetNumberOfTuples(points.shape[0])
        for i in range(points.shape[0]):
            vcolors.SetTuple3(i, colors[i, 0], colors[i, 1], colors[i, 2])
        vpoly.GetPointData().SetScalars(vcolors)

    vcells = vtk.vtkCellArray()

    for i in range(points.shape[0]):
        vcells.InsertNextCell(1)
        vcells.InsertCellPoint(i)

    vpoly.SetVerts(vcells)

    return vpoly

def create_pointcloud_actor(points, colors=None):
    """Creates a vtkActor with the point cloud from numpy arrays

    points: numpy.ndarray
        pointcloud with shape (n,3)

    colors: numpy.ndarray
        uint8 array with colors for each point. shape is (n,3)

    Returns vtkActor object
    """
    vpoly = create_pointcloud_polydata(points, colors)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(vpoly)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetPointSize(5)

    return actor


def visualize_prediction(pointcloud, colors, poses=None, renwin=None):
    """ render point cloud and cameras """

    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0, 0, 0)

    pointcloud_actor = create_pointcloud_actor(points=pointcloud, colors=colors)
    pointcloud_actor.GetProperty().SetPointSize(2)
    renderer.AddActor(pointcloud_actor)

    for pose in poses:
        R, t = pose[:3, :3], pose[:3, 3]
        cam_actor = create_camera_actor(R, t)
        cam_actor.GetProperty().SetColor((255, 255, 0))
        renderer.AddActor(cam_actor)

    camera = vtk.vtkCamera()
    camera.SetPosition((1, -1, -2))
    camera.SetViewUp((0, -1, 0))
    camera.SetFocalPoint((0, 0, 2))

    renderer.SetActiveCamera(camera)
    renwin = vtk.vtkRenderWindow()

    renwin.SetWindowName("Point Cloud Viewer")
    renwin.SetSize(800, 600)
    renwin.AddRenderer(renderer)

    # An interactor
    interactor = vtk.vtkRenderWindowInteractor()
    interstyle = vtk.vtkInteractorStyleTrackballCamera()
    interactor.SetInteractorStyle(interstyle)
    interactor.SetRenderWindow(renwin)

    # Render and interact
    renwin.Render()
    interactor.Initialize()
    interactor.Start()
