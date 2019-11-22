import numpy as np
import logging

from geometrySettings import GeometrySettings

logger = logging.getLogger(__name__)


class Camera:
    ''' Class representing the camera object. 
    cx and cy follow openCV conventions (wrt top-left corner). f, cx and cy are in pixels.
    
    Attributes:
        intrinsics ({})                         : camera intrinsics {'f': float, 'cx': float, 'cy': float, 'k1': float, 'k2': float, 'k3' :float, 
                                                    'p1':float, 'p2':float, 'width': int, 'height': int}
        R (np.matrix)                           : rotation matrix according to the world coordinate system (.out format)
        t (np.matrix)                           : world system coordinates wrt the camera center (.out format). It's general name is translation vector
        points3D (list(int))                    : ids of of visible points3D
        type (CameraType(Enum))                 : type of the camera (terrestrial (TER) or aeral (UAV)) 
    '''
    def __init__(self, id):
        self.__id = id
        self.__intrinsics = {'f': None, 'cx': None, 'cy': None, 'k1': None, 'k2': None, 'p1': None, 'p2': None, 'k3': None, 'width' : None, 'height' : None}
        self.__R = None
        self.__t = None
        self.__points3D = []
        self.__type = GeometrySettings.CameraType.GENERIC
        
    def __repr__(self):
        return 'Camera ID: {}, I: {}, R: {}, t: {}, type: {}'.format(self.__id, self.__intrinsics, self.__R.tolist(), self.__t.tolist(), self.__type)


    ''' ************************************************ Setters ************************************************ '''
    def set_focal(self, f):
        ''' Set the camera focal.

            Attributes:
                f (float)   :   focal length
        '''
        self.__intrinsics['f'] = f
    
    def set_image_resolution(self, width, height):
        ''' Set the image resolution (width x height).

            Attributes:
                width   (int)   :   image width
                height  (int)   :   image height
        '''
        if width <= 0 or height <= 0:
            logger.critical('{} x {} are invalid width and height values'.format(width, height))
            exit(1)
        self.__intrinsics['width'] = width
        self.__intrinsics['height'] = height

    def set_camera_center(self, cx, cy):
        ''' Set the camera center. Opencv format: in pixels and wrt top left corner.

            Attributes:
                cx (float)  :   x coordinate of the camera center
                cy (float)  :   y coordinate of the camera center
        '''
        self.__intrinsics['cx'] = float(cx)
        self.__intrinsics['cy'] = float(cy)

    def set_radial_distortion_params(self, k1, k2, k3 = None):
        ''' Set radial distortion parameters. 

            Attributes:
                k1 (float)  :   first order parameter
                k2 (float)  :   second order parameter
                k3 (float)  :   third order parameter (optional)
        '''
        self.__intrinsics['k1'] = float(k1)
        self.__intrinsics['k2'] = float(k2)
        if k3:
            self.__intrinsics['k3'] = float(k3) 

    def set_tangential_distortion_params(self, p1, p2):
        ''' Set the tangential distortion parameters.

            Attributes:
                p1 (float)  :   first order parameter
                p2 (float)  :   second order parameter
        '''
        self.__intrinsics['p1'] = float(p1)
        self.__intrinsics['p2'] = float(p2)

    def set_rotation_matrix(self, R, format):
        ''' Set the rotation matrix.

            Attributes:
                R (np.matrix)                   :   rotation matrix
                format (RotationMatrixType)     :   format of the rotation matrix
        '''
        if not R.shape == (3,3):
            logger.critical('{} is an invalid rotation matrix shape'.format(R.shape))
            exit(1)
        
        if format == GeometrySettings.RotationMatrixType.BUNDLER_OUT:
            self.__R = R
        elif format == GeometrySettings.RotationMatrixType.BLOCK_EXCHANGE:
            logger.critical('Block Exchange format not yer supported')                          # TODO
            exit(1)
        else:
            logger.critical('Unknown rotation matrix type')
            exit(1)

    def set_translation_vector(self, t, format):
        ''' Set the translation vector

            Attributes:
                t (np.matrix)                   :   translation vector
                format (TranslationVectorType)  :   format of the translation vector
        '''
        if not t.shape == (3,1):
            logger.critical('{} is an invalid translation vector shape'.format(t.shape))
            exit(1)
        
        if format == GeometrySettings.TranslationVectorType.BUNDLER_OUT:
            self.__t = t
        elif format == GeometrySettings.TranslationVectorType.BLOCK_EXCHANGE:
            logger.critical('Block Exchange format not yet supported')                          # TODO
            exit(1)
        else:
            logger.critical('Unknown translation vector type')
            exit(1)

    def add_visible_point3D(self, p3D_id):
        ''' Add a visible point3D in this camera

            Attributes:
                p3D_id (int)   : id of the point3D   
        '''
        self.__points3D.append(p3D_id)


    ''' ************************************************ Getters ************************************************ '''
    def get_id(self):
        return self.__id

    def get_image_resolution(self):
        ''' Return image resolution (width and height).

            Attributes:
                width, height   (int, int)  :   width and height
        '''
        if not self.__intrinsics['width'] or not self.__intrinsics['height']:
            logger.critical('Image resolution not set')
            exit(1)
        return self.__intrinsics['width'], self.__intrinsics['height']

    def get_camera_matrix_opencv(self):
        ''' Return a camera matrix with the instrinsic parameters using opencv format

            Return:
                _ (np.matrix)   :   3 x 3 matrix in the opencv format
        '''
        if not self.__intrinsics['cx'] or not self.__intrinsics['cy']:
            logger.critical('Camera {} has no camera center'.format(self.__id))
            exit(1)
        if not self.__intrinsics['f']:
            logger.critical('Camera {} has no focal'.format(self.__id))
            exit(1)

        return np.matrix([
                [self.__intrinsics['f'], 0, self.__intrinsics['cx']], 
                [0, self.__intrinsics['f'], self.__intrinsics['cy']], 
                [0, 0, 1]
            ], dtype=np.float)

    def get_radial_distortion_params(self):
        ''' Return radial distortion parameters.

            Return:
                k1, k2, k3   (float, float, float)  :   radial distortion parameters 
        '''
        if self.__intrinsics['k1'] and self.__intrinsics['k2'] and self.__intrinsics['k3']:
            return self.__intrinsics['k1'], self.__intrinsics['k2'], self.__intrinsics['k3']
        else:
            logger.critical('Radial distortion parameters are not set')
            exit(1)

    def get_tangential_distortion_params(self):
        ''' Return tangential distortion parameters

            Return:
                p1, p2 (float, float)   :   tangential distortion parameters
        '''
        if self.__intrinsics['p1'] and self.__intrinsics['p2']:
            return self.__intrinsics['p1'], self.__intrinsics['p2']
        else:
            logger.critical('Tangential distortion parameters are not set')
            exit(1)

    def distortion_vector_opencv(self):
        ''' Returns the instrinsic distortion attributes in a distortion vector following the conventions of opencv
        '''
        return np.array([self.__intrinsics['k1'], self.__intrinsics['k2']], dtype=np.float)

    def get_rotation_matrix(self, format):
        ''' Return the rotation matrix in the specified format

            Attributes:
                format (RotationMatrixType) :   format of the returned rotation matrix

            Return:
                R (np.matrix)   :   rotation matrix of this camera in the requested format
        '''
        if format == GeometrySettings.RotationMatrixType.BUNDLER_OUT:
            return np.copy(self.__R)
        elif format == GeometrySettings.RotationMatrixType.BLOCK_EXCHANGE:
            R_be = np.matrix([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).T.dot(self.__R)
            return R_be
        else:
            logger.critical('{} is an invalid rotation matrix format'.format(format))
            exit(1)

    def get_translation_vector(self, format):
        ''' Return the translation vector in the specified format

            Attributes:
                format (TranslationVectorType) :   format of the returned translation vector

            Return:
                t (np.matrix)   :   translation vector of this camera in the requested format
        '''
        if format == GeometrySettings.TranslationVectorType.BUNDLER_OUT:
            return np.copy(self.__t)                                                        # World coordinate system wrt camera center
        elif format == GeometrySettings.TranslationVectorType.BLOCK_EXCHANGE:
            t_be = np.negative(self.__R).T.dot(self.__t)                                            # Camera center wrt the world coordinate system
            return t_be
        else:
            logger.critical('{} is an invalid translation vector format'.format(format))
            exit(1)


    ''' ************************************************ Modifiers ************************************************ '''
    def remove_point3D(self, p3D_id):
        ''' Removes a point3D from this camera

            Attributes:
                p3D_id (int)    : id of the point3D to remove
        '''
        try:
            self.__points3D.remove(p3D_id)
        except ValueError:
            logger.critical('Point3D {} not in camera {}'.format(p3D_id, self.__id))
            exit(0)


    