import os
import logging

logger = logging.getLogger(__name__)

class OutReader():
    def __init__(self, path, lines_per_camera = 5, lines_per_point3D = 3):
        ''' Costructor.

            Attributes:
                path                (string)    :   path to the file to read
                lines_per_camera    (int)       :   number of lines used in the file to store the information of a single camera
                lines_per_point3D   (int)       :   number of lines used in the file to store the information of a single point3D
        '''
        self.lines_per_camera = lines_per_camera
        self.lines_per_point3D = lines_per_point3D
        self.lines = []
        logger.debug('Number of lines for each camera: {}'.format(self.lines_per_camera))
        logger.debug('Number of lines for each point3D: {}'.format(self.lines_per_point3D))

        self.__read_file(path)

    def __read_file(self, path):
        ''' Load the file lines.

            Attributes:
                path    (string)    :   path to the file to read
        '''
        if not os.path.isfile(path):
            logger.critical('File "{}" does not exist'.format(path))
            exit(1)

        try:
            with open(path) as f:
                for line in f.readlines():
                    self.lines.append(line.strip())  
        except IOError:
            logger.critical('File "{}" is not readable'.format(path))
            exit(1)

    
    def get_file_content(self):
        ''' Parse the file lines and store the data in two list of dictionaries.
            Each camera is a dictionary containing camera intrinsics (cam_intr)[f, k1, k2], rotation matrix (R) and camera translation (t).
            Each point3D is a dictionary containing position (xyz), color (rgb) and 2D observations (obs)

            Return:
                cameras ([{string -> _}])   : list of cameras {'cam_intr' : string, 'R' : [[string]], 't' : string}
                points3D ([{string -> _}])  : list of points3D {'xyz' : [string], 'rgb' : [string], 'obs' : string}
        '''
        cameras, points3D = [], []

        # Get the number of cameras and the number of 3Dpoints
        tokens = self.lines[1].strip().split(' ')
        num_cameras = int(tokens[0]) 
        num_points3D = int(tokens[1])
        logger.debug('Espected number of cameras: {}'.format(num_cameras))
        logger.debug('Espected number of points3D: {}'.format(num_points3D))
        
        # Read the cameras
        for line_index, line in enumerate(self.lines[2:(num_cameras * self.lines_per_camera)+2]):
            tokens = line.strip().split(' ')
            
            if line_index % self.lines_per_camera == 0:             # Beginning camera string <f> <k1> <k2>
                camera = {'cam_intr': tokens, 'R': [], 't': []}
            elif line_index % self.lines_per_camera < 4:            # <R>
                camera['R'].append(tokens)                  
            elif line_index % self.lines_per_camera == 4:           # End of the camera string, <t1> <t2> <t3>  
                camera['t'] = tokens
                cameras.append(camera)

        # Read the points3D
        for line_index, line in enumerate(self.lines[(num_cameras * self.lines_per_camera)+2:]):
            tokens = line.strip().split(' ')

            if line_index % self.lines_per_point3D == 0:
                point3D = {'xyz': tokens, 'rgb': [], 'obs': []}
            elif line_index % self.lines_per_point3D == 1:
                point3D['rgb'] = tokens
            elif line_index % self.lines_per_point3D == 2:
                point3D['obs'] = tokens
                points3D.append(point3D)
        
        assert len(cameras) == num_cameras, 'Espected cameras: {},  found: {}'.format(num_cameras, len(cameras))
        assert len(points3D) == num_points3D, 'Espected points3D: {},  found: {}'.format(num_points3D, len(points3D))
        logger.info('Found {} cameras and {} points3D'.format(len(cameras), len(points3D)))
        
        return cameras, points3D