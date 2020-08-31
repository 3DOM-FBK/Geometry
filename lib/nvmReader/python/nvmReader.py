import os
import logging

logger = logging.getLogger(__name__)

class NvmReader():
    def __init__(self, path):
        ''' Constructor.

            Attributes:
                path    (string)    :   path to the file to read
        '''
        self.lines = []
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
            logger.info('Read {} lines from {}'.format(len(self.lines), path))
        except IOError:
            logger.critical('File "{}" is not readable'.format(path))
            exit(1)


    def get_file_content(self):
        ''' Parse the file lines and store the data in two list of dictionaries.
            Each camera is a dictionary containing camera intrinsics (cam_intr), quaterions (q) and camera center (cc).
            Each point3D is a dictionary containing position (xyz), color (rgb) and 2D observations (obs)

            Return:
                cameras ([{string -> _}])    : list of camera {'cam_intr' : string, 'q' : [[string]], 'cc' : string}
                points3D ([{string -> _}])   : list of point3D {'xyz' : [string], 'rgb' : [string], 'obs' : string}
        '''
        cameras, points3D = [], []

        # Get the number of cameras and the number of 3Dpoints  
        num_cameras = int(self.lines[2])
        num_points3D = int(self.lines[num_cameras + 4])
        logger.debug('Espected number of cameras: {}'.format(num_cameras))
        logger.debug('Espected number of points3D: {}'.format(num_points3D))

        # Read the cameras
        for line in self.lines[3: num_cameras + 3]:
            tokens = line.strip().split(' ')
            
            camera = {'cam_intr': [], 'q': [], 'cc': []}
            camera['cam_intr'].append(tokens[1])
            camera['cam_intr'].append(tokens[9])
            camera['q'] += tokens[2:6]
            camera['cc'] += tokens[6:9]
            cameras.append(camera)
        
        # Read the points3D
        for line in self.lines[num_cameras + 5:]:   
            tokens = line.strip().split(' ')
            
            point3D = {'xyz': [], 'rgb': [], 'obs': []}
            point3D['xyz'] += tokens[0:3]
            point3D['rgb'] += tokens[3:6]
            point3D['obs'] += tokens[6:]
            points3D.append(point3D)
        
        assert len(cameras) == num_cameras, 'Espected cameras: {},  found: {}'.format(num_cameras, len(cameras))
        assert len(points3D) == num_points3D, 'Espected points3D: {},  found: {}'.format(num_points3D, len(points3D))
        logger.info('Found {} cameras and {} points3D'.format(len(cameras), len(points3D)))

        return cameras, points3D

