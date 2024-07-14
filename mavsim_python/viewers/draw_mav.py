# Copy and paste draw_spacecraft.py here and then edit to draw a UAV instead of the spacecraft.

import numpy as np
import pyqtgraph.opengl as gl
from tools.rotations import euler_to_rotation
from tools.drawing import rotate_points, translate_points, points_to_mesh


class DrawMav:
    def __init__(self, state, window, scale=10):
        """
        Draw the Mav.

        The input to this function is a (message) class with properties that define the state.
        The following properties are assumed:
            state.north  # north position
            state.east  # east position
            state.altitude   # altitude
            state.phi  # roll angle
            state.theta  # pitch angle
            state.psi  # yaw angle
        """
        self.unit_length = scale
        mav_position = np.array([[state.north], [state.east], [-state.altitude]])  # NED coordinates
        # attitude of mav as a rotation matrix R from body to inertial
        R_bi = euler_to_rotation(state.phi, state.theta, state.psi)
        # convert North-East Down to East-North-Up for rendering
        self.R_ned = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        # get points that define the non-rotated, non-translated spacecraft and the mesh colors
        self.mav_points, self.mav_index, self.mav_meshColors = self.get_mav_points()
        self.mav_body = self.add_object(
            self.mav_points,
            self.mav_index,
            self.mav_meshColors,
            R_bi,
            mav_position)
        window.addItem(self.mav_body)  # add spacecraft to plot     

    def update(self, state):
        mav_position = np.array([[state.north], [state.east], [-state.altitude]])  # NED coordinates
        # attitude of mav as a rotation matrix R from body to inertial
        R_bi = euler_to_rotation(state.phi, state.theta, state.psi)
        self.mav_body = self.update_object(
            self.mav_body,
            self.mav_points,
            self.mav_index,
            self.mav_meshColors,
            R_bi,
            mav_position)

    def add_object(self, points, index, colors, R, position):
        rotated_points = rotate_points(points, R)
        translated_points = translate_points(rotated_points, position)
        translated_points = self.R_ned @ translated_points
        # convert points to triangular mesh defined as array of three 3D points (Nx3x3)
        mesh = points_to_mesh(translated_points, index)
        object = gl.GLMeshItem(
            vertexes=mesh,  # defines the triangular mesh (Nx3x3)
            vertexColors=colors,  # defines mesh colors (Nx1)
            drawEdges=True,  # draw edges between mesh elements
            smooth=False,  # speeds up rendering
            computeNormals=False)  # speeds up rendering
        return object

    def update_object(self, object, points, index, colors, R, position):
        rotated_points = rotate_points(points, R)
        translated_points = translate_points(rotated_points, position)
        translated_points = self.R_ned @ translated_points
        # convert points to triangular mesh defined as array of three 3D points (Nx3x3)
        mesh = points_to_mesh(translated_points, index)
        object.setMeshData(vertexes=mesh, vertexColors=colors)
        return object

    def get_mav_points(self):
        """"
            Points that define the spacecraft, and the colors of the triangular mesh
            Define the points on the spacecraft following information in Appendix C.3
        """
        
        # TODO: Need to update these values.
        fuse_h = 2
        fuse_w = 1
        fuse_l1 = 2
        fuse_l2 = 1
        fuse_l3 = 3
        wing_l = 1
        wing_w = 3
        tail_h = 1
        tail_wing_l = 1
        tail_wing_w = 3

        # points are in XYZ coordinates
        #   define the points on the spacecraft according to Appendix C.3
        points = self.unit_length * np.array([
            [fuse_l1, 0, 0], 
            [fuse_l2, fuse_w/2, -fuse_h/2],
            [fuse_l2, -fuse_w/2, -fuse_h/2],
            [fuse_l2, -fuse_w/2, fuse_h/2],
            [fuse_l2, fuse_w/2, fuse_h/2],
            [-fuse_l3, 0, 0],
            [0, wing_w/2, 0],
            [-wing_l, wing_w/2, 0],
            [-wing_l, -wing_w/2, 0],
            [0, -wing_w/2, 0],
            [-(fuse_l3-tail_wing_l), tail_wing_w/2, 0],
            [-fuse_l3, tail_wing_w/2, 0],
            [-fuse_l3, -tail_wing_w/2, 0],
            [-(fuse_l3-tail_wing_l), -tail_wing_w/2, 0],
            [-(fuse_l3-tail_wing_l),0, 0],
            [-fuse_l3, 0, -tail_h]
            ]).T
        
        # point index that defines the mesh
        index = np.array([
            [2, 1, 0],
            [3, 2, 0],
            [4, 3, 0],
            [0, 1, 4],
            [1, 2, 5],
            [2, 3, 5],
            [3, 4, 5],
            [5, 4, 1],  # fuselage
            [6, 9, 8],
            [8, 7, 6],  # wing
            [10, 13, 12],
            [12, 11, 10],  # tailwing
            [14, 15, 5]   # tail
            ])
        #   define the colors for each face of triangular mesh
        red = np.array([1., 0., 0., 1])
        green = np.array([0., 1., 0., 1])
        blue = np.array([0., 0., 1., 1])
        yellow = np.array([1., 1., 0., 1])
        meshColors = np.empty((13, 3, 4), dtype=np.float32)
        meshColors[:] = green # nose-top

        return points, index, meshColors

