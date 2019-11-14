import numpy as np
import logging

logger = logging.getLogger()

class Point2D:
    ''' Class representing an image point (2D) 

    Attributes:
        xy (numpy array)                    : image coordinates according to openCV convention (relative to the top-left corner)
        keypoint_index (tuple(int))         : keypoint index of the this 2D point in the camera. first: camera id, second: keypoint index
        p3D_id (int)                        : corresponding 3D point id
    '''

    def __init__(self, id):
        self.__id = id
        self.__xy = None
        self.__keypoint_index = (-1,-1)
        self.__p3D_id = -1
    
    def __repr__(self):
        return 'P2D id: {}, xy: {}, kp_index: {}, p3D_id: {}'.format(self.__id, self.__xy.tolist(), self.__keypoint_index, self.__p3D_id)
  

    ''' ************************************************ Setters ************************************************ '''
    def set_coordintes(self, xy):
        ''' Set the image coordinates of the point2D 

            Attributes:
                xy (np.matrix(float))   :   image coordinates (x,y)  
        '''
        if not xy.shape == (2,1):
            logger.critical('{} is an invalid point2D coordinate shape'.format(xy.shape))
            exit(1)
        self.__xy = xy     

    def set_keypoint_index(self, cam_id, kp_index):
        ''' Set the keypoint index of the point2D in the correspondig camera

            Attributes:
                cam_id (int)    :   camera id in which the point2D was extracted
                kp_index (int)  :   keypoint index in the camera image features
        ''' 
        if cam_id < 0:
            logger.warning('Negative id for camera id: {}'.format(cam_id))
        if kp_index < 0:
            logger.warning('Negative id for keypoint index: {}'.format(kp_index)) 
        self.__keypoint_index = (cam_id, kp_index)
 
    def set_point3D(self, p3D_id):
        ''' Set the corresponding point3D id.

            Attributes:
                p3D_id (int)    :   id of the corresponding point3D
        '''
        if p3D_id < 0:
            logger.warning('Negative id for point3D id: {}'.format(p3D_id)) 
        self.__p3D_id = p3D_id


    ''' ************************************************ Getters ************************************************ '''
    def get_coordinates(self, homogenous = False):
        ''' Return the image coordinates (x,y)

            Attributes:
                homogenous (bool)   :   return coordinates in homogeneus coordinates (optional)

            Return:
                _ (np.matrix)    :   2D coordinates. (2,1) or (3,1) (if homogeneous)
        '''
        return np.copy(self.__xy)
