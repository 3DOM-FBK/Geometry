import logging
import numpy as np
from itertools import chain

import camera as Camera
from geometrySettings import GeometrySettings

logger = logging.getLogger()


class Point3D:
    ''' Class representing an object point (3D)

    Attributes:
        xyz (numpy matrix)              :   x,y and z coordinates
        rgb (numpy array)               :   red, green and blue
        observations ({})               :   observations of the this point3D in the cameras. Key: cam_id, value: Point2D id
        features({})                    :   dictionary of the features {'reprojection_error' : float, 'max_angle' : float}
    '''
    def __init__(self, id):
        self.__id = id
        self.__xyz = None
        self.__rgb = None
        self.__observations = {}
        self.__features = {}

    def __repr__(self):
        return 'P3D id: {}, xyz: {}, rgb: {}, observations: {}, features: {}'.format(self.__id, self.__xyz.tolist(), self.__rgb, self.__observations, self.__features)
    

    ''' ************************************************ Setters ************************************************ '''
    def set_coordinates(self, xyz):
        ''' Set the point3D coordinates in the world coordinate system.

            Attributes:
                xyz (np.matrix(float))     :   point coordinates (X Y Z) with shape 3x1
        ''' 
        if not xyz.shape == (3,1):
            logger.critical('{} is an invalid point3D coordinate shape'.format(xyz.shape))
            exit(1)
        self.__xyz = xyz

    def set_color_rgb(self, rgb):
        ''' Set the point3D color in rgb

            Attributes:
                rgb (np.array(int))     :   red, green and blue components with shape 1x3
        '''
        if not rgb.shape == (3,):
            logger.critical('{} is an invalid point3D rgb shape'.format(rgb.shape))
            exit(1)
        self.__rgb == rgb

    def set_observation(self, cam_id, p2D_id):
        ''' Set observation (camera and corresponding point2D).

            Attributes:
                cam_id (int)    :   camera id
                p2D_id (jnt)    :   point2D id
        '''
        if cam_id < 0:
            logger.warning('Negative id for camera id: {}'.format(cam_id))
        if p2D_id < 0:
            logger.warning('Negative id for point2D id: {}'.format(p2D_id)) 
        self.__observations[cam_id] = p2D_id

    def set_mean_reprojection_error(self, reprojection_error):
        ''' Set the mean reprojection error of the cameras seing this point3D.

            Attributes:
                re (float)  : mean reprojection error
        '''
        if reprojection_error < 0:
            logger.warning('Point3D {} has negative reprojection error {}'.format(self.__id, reprojection_error))
        
        self.__features['reprojection_error'] = reprojection_error

    def set_max_intersection_angle(self, max_angle):
        ''' Set the maximum intersection angle of the cameras seing this point3D.

            Attributes:
                max_angle (float)   : maximum intersection angle
        '''
        if max_angle < 0:
            logger.warning('Negative maximum intersection angle {} for p3D {}'.format(max_angle, self.__id))
  
        self.__features['max_angle'] = max_angle
    
    def set_feature(self, name, value):
        ''' Set a generic feature of the point3D.

            Attributes:
                name (string)       :   name of the feature
                value (_)           :   value of the feature
        '''
        self.__features[name] = value


    ''' ************************************************ Getters ************************************************ '''
    def get_coordinates(self, homogeneous = False):
        ''' Return the 3D coordinates of the point3D.

            Attributes:
                homogenous (bool)   :   return coordinates in homogeneus coordinates (optional)
            
            Return:
                _ (np.matrix)       :   3D coordinates. 3x1 or 4x1 (if homogeneous)
        '''
        if homogeneous:
            return np.copy(np.append(self.__xyz, [[1]], axis=0))
        else:
            return np.copy(self.__xyz)
    
    def get_coordinates_as_string(self):
        ''' Return the 3D cordinates as a string "x y z"

            Return
                xyz (string)    :   coordinates as string
        '''
        xyz_flattened = list(chain.from_iterable(self.__xyz.tolist()))
        return " ".join([str(v) for v in xyz_flattened])

    def get_cameras_seing_me(self):
        ''' Return the ids of the cameras seing this point3D.

            Return:
                cam_ids [int]   :   ids of the observing cameras
        '''
        if len(self.__observations) == 0:
            logger.error('No cameras are seing point3D: {}'.format(self.__id))
            return None
        
        return [cam_id for cam_id,_ in self.__observations.items()]
        
    def get_observation (self, cam_id):
        ''' Return the id of the point2D observed in the given camera.

            Attributes:
                cam_id (int)    :   camera id

            Return:
                p2D_id (int)    :    id of the point2D (observation)
        '''
        if cam_id not in self.__observations:
            logger.critical('Camera id {} not in the observations of p3D with id: {}'.format(cam_id, self.__id))
        
        return self.__observations[cam_id] 

    def get_mean_reprojection_error(self):
        ''' Return the mean reprojection error

            Return:
                _ (float)   : mean reprojection error
        '''
        if 'reprojection_error' not in self.__features or not self.__features['reprojection_error']:
            logger.error('No reprojection error value for point3D {}'.format(self.__id))
            return
        
        return self.__features['reprojection_error']
    
    def get_max_intersection_angle(self):
        ''' Return the max intersection angle

            Return:
                _ (float)   : max intersection angle
        '''
        if 'max_angle' not in self.__features or not self.__features['max_angle']:
            logger.error('No max intersection angle value for point3D {}'.format(self.__id))
            return
        
        return self.__features['max_angle']
    
    def get_feature(self, feature):
        ''' Get a generic feature of the point3D.

            Attributes:
                feature (string)    :   name of the feature
        '''
        if not feature in self.__features.keys():
            logger.critical('Point3D {} has no feature {}'.format(self.__id, feature))
            exit(1)
        
        return self.__features['feature']


    ''' ************************************************ Modifiers ************************************************ '''
    def remove_observation(self, cam_id):
        ''' Remove an observation ({cam_id -> point2D->id})

            Attributes:
                cam_id (int)    : id of the camera (key of the observation)
        '''       
        try:
            del self.__observations[cam_id]
        except KeyError:
            logger.warning('Point3D {} is not observed in camera {}'.format(self.__id, cam_id))

    #def set_point3D_type(self, cameras):
        ''' Sets the point type (TER, UAV, TER-UAV) based on the type of cameras observing this point.

            Attributes:
                cameras (dict[int->Camera])     : Dictionary containing all the cameras
        '''
        '''observations = [0] * 2               # observations[0] are TER observations, observations[1] are the UAV ones
        
        # Fill the observations 
        for camera_id, _ in self.__observations.items():
            if cameras[camera_id].type == Camera.CameraType.TER:
                observations[0] += 1
            elif cameras[camera_id].type == Camera.CameraType.UAV:
                 observations[1] += 1
        
        # Use the observations to decide the point3D type
        if observations[0] > 0 and observations[1] > 0:
            self.type = PointType.TER_UAV
        elif observations[0] > 0 and observations[1] == 0:
            self.type = PointType.TER
        elif observations[0] == 0 and observations[1] > 0:
            self.type = PointType.UAV'''