import os
import time
import logging
import numpy as np 
from sys import path

# Add submodules
path.insert(0, os.path.join(os.path.dirname(__file__), 'lib/outReader/python/'))
path.insert(0, os.path.join(os.path.dirname(__file__), 'lib/nvmReader/python/'))

from camera import Camera
from point3D import Point3D
from point2D import Point2D
from outReader import OutReader
from nvmReader import NvmReader
from geometrySettings import GeometrySettings

logger = logging.getLogger()


class Geometry:
    ''' Representation of a sparse reconstruction.

        Attributes:
            cameras {int -> Camera}         :   cameras/images of the reconstruction
            points2D {int -> Point2D}       :   image observations
            points3D {int -> Point3D}       :   points in the object space (3D)
            features {string -> [string]    :   computed and imported features for points3D, points2D and cameras
    '''
    def __init__(self):
        self.__cameras = {}
        self.__points3D = {}
        self.__points2D = {}
        
        self.__features = {'point2D' : [], 'point3D': [], 'camera' : []}


    ''' ************************************************ Import ************************************************ '''
    def load_reconstruction(self, filepath):
        ''' Load the cameras, the 3D points, and the 2D points from the input file
            Supported types: .nvm and .out

            Attributes:
                filepath (string)   :   path to the input file.
        '''
        filetype =  filepath.split('.')[-1].lower()
        logger.info('Input: {}'.format(filepath))

        if filetype == GeometrySettings.SupportedInputFileType.OUT.name.lower():
            self.__load_out(filepath)
        elif filetype == GeometrySettings.SupportedInputFileType.NVM.name.lower():
            self.__load_nvm(filepath)
        else:
            logger.critical('Error: file type "{}" not supported'.format(filetype))     
            exit(1)      
        
        logger.info('Data loaded. Cameras: {}, Points3D: {}, Points2D: {}'.format(len(self.__cameras), len(self.__points3D), len(self.__points2D)))

    def load_full_camera_intrinsics(self, filepath, format):
        ''' Load fx, fy, cx, cy, k1, k2, k3, p1, p2, width and height from a .txt file.  
            File format: <cam_id> <f> <cx> <cy> <k1> <k2> <k3> <p1> <p2> <width> <height>
            <cam_id> should be given considering the order of the cameras in the .out/.nvm files.
            Values will be stored using the OPENCV format. Fx and fy are considered equal.

            Attributes:
                filepath (string)               :   path to the file 
                format (InstriscsFormatType)    :   format of the file intrinsics parameters
        '''
        if not os.path.exists(filepath):
            logger.critical('File {} does not exists'.format(filepath))
            exit(1)
        if format != GeometrySettings.InstriscsFormatType.OPENCV and format != GeometrySettings.InstriscsFormatType.METASHAPE:
            logger.critical('Unknown instrinsic format')
            exit(1)

        try:
            with open (filepath, 'r') as f:
                file_lines = f.readlines()
            if len(file_lines) < len(self.__cameras):
                logger.critical('intrinsics not specified for all the cameras. Expected cameras: {}, found: {}. Check your file'.format(len(self.__cameras), len(file_lines)))
                exit(1)
        except IOError:
            logger.critical('File "{}" is not readable'.format(filepath))
            exit(1)

        for line in file_lines: 
            tokens = line.replace('\t', ' ').strip().split(' ')
            if len(tokens) < 11:
                logger.critical('Invalid instrinsic file format found at line {}. ' \
                    'Supported file format: <cam_id> <f> <cx> <cy> <k1> <k2> <k3> <p1> <p2> <width> <height>'.format(line))
                exit(1)
            
            cam_id = int(tokens[0])
            self.__cameras[cam_id].set_focal(f = float(tokens[1]))
            self.__cameras[cam_id].set_radial_distortion_params(k1 = float(tokens[4]), k2 = float(tokens[5]), k3 = float(tokens[6]))
            self.__cameras[cam_id].set_tangential_distortion_params(p1 = float(tokens[7]), p2 = float(tokens[8]))
            self.__cameras[cam_id].set_image_resolution(width = int(tokens[9]), height = int(tokens[10]))

            if format == GeometrySettings.InstriscsFormatType.OPENCV:
                self.__cameras[cam_id].set_camera_center(cx = float(tokens[2]), cy = float(tokens[3]))
            elif format == GeometrySettings.InstriscsFormatType.METASHAPE:                      # Camera center in metashape is specified as an offset.
                width, height = self.__cameras[cam_id].get_image_resolution()
                cx_opencv = float(tokens[2]) + (width / 2)                                      # Convert it into coordinates wrt top-left corner
                cy_opencv = float(tokens[3]) + (height / 2)
                self.__cameras[cam_id].set_camera_center(cx = cx_opencv, cy = cy_opencv)

            logger.debug('Updated intrinsics: {}'.format(self.__cameras[cam_id]))
        logger.info('Loaded full intrinsics of {} cameras from {} expecting {} format'.format(len(file_lines), filepath, format.name))

    def __load_out(self, filepath):
        ''' Load the content of a .out file

            Attributes:
                filepath (string)   :   path to the .out file
        '''
        # Get the content of the file
        reader = OutReader(filepath) 
        cameras_json, points3D_json = reader.get_file_content() 
        del reader

        # Create the cameras
        for cam_id, cam_data in enumerate(cameras_json):
            camera = Camera(cam_id)
            
            camera.set_focal(float(cam_data['cam_intr'][0]))                                            # Set focal 
            camera.set_radial_distortion_params(k1=float(cam_data['cam_intr'][1]),                      # Set radial distortion
                k2=float(cam_data['cam_intr'][2]))

            camera.set_rotation_matrix(R = np.matrix(cam_data['R'], dtype = np.float),                      # Set rotation matrix 
                format = GeometrySettings.RotationMatrixType.BUNDLER_OUT)                 
                                    
            camera.set_translation_vector(t = np.matrix(cam_data['t'], dtype = np.float).reshape(-1,1),     # Set the translation vector
                format = GeometrySettings.TranslationVectorType.BUNDLER_OUT) 
            
            logger.debug('New {}'.format(camera))
            self.__cameras[cam_id] = camera
        
        # Create points3D and points2D (observations)
        p2D_id = 0
        for p3D_id, p3D_data in enumerate(points3D_json):
            p3D = Point3D(p3D_id)

            p3D.set_coordinates(np.matrix(p3D_data['xyz'], dtype = np.float).reshape(-1,1))           # Position (column vector) and color (row vector)
            p3D.set_color_rgb(np.array(p3D_data['rgb'], dtype = np.int))

            observations = self.__parse_observation_string(p3D_data['obs'], 'out')                  # 2D observations: camera-id keypoint-index x y
            for observation in observations:
                p2D = Point2D(p2D_id)
                p2D.set_keypoint_index(int(observation['cam_id']), int(observation['kp_index']))
                p2D.set_coordintes(np.matrix([float(observation['x']), float(observation['y'])]).reshape(-1,1))
                p2D.set_point3D(int(p3D_id))

                p3D.set_observation(self.__cameras[observation['cam_id']].get_id(), p2D_id)        # Add 2D observation id to the point3D
                self.__cameras[observation['cam_id']].add_visible_point3D(p3D_id)                  # Add point3D to the visible points of the camera
        
                logger.debug('New {}'.format(p2D))
                self.__points2D[p2D_id] = p2D
                p2D_id += 1 
            
            logger.debug('New {}'.format(p3D))
            self.__points3D[p3D_id] = p3D

    def __load_nvm(self, filepath):
        ''' Load the content of a .nvm file

            Attributes:
                filepath (string)   :   path to the .nvm file
        '''
        # Get the content of the file
        reader = NvmReader(filepath) 
        cameras_json, points3D_json = reader.get_file_content()
        del reader

        # Create the cameras
        for cam_id, cam_data in enumerate(cameras_json):
            pass
        
        # Create the points3D and points2D (observations)
        for p3D_id, p3D_data in enumerate(points3D_json):
            pass

    def __parse_observation_string(self, observation_string, filetype):
        ''' Parse the observation string. 

            Attributes:
                observation_string (string)     :   observation string in the filetype format
                filetype (string)               :   use the filetype to parse correcty the observation string 
            
            Returns:
                res ([{}])                      : list of observations with format {cam_id : int, kp_index : int, x : float, y : float }
        '''
        num_observations = int(observation_string[0])  

        observations = []
        if filetype == 'out':
            for i in range(1, 4 * num_observations, 4):
                observations.append({'cam_id' : int(observation_string[i]), 'kp_index' : int(observation_string[i+1]),
                    'x' : float(observation_string[i+2]), 'y' : float(observation_string[i+3])})
        elif filetype == 'nvm':
            logger.critical('Parsing nvm observation strings is not yet supported')
            exit(1)
        else:   
            logger.critical('Error parsing observation string, filetype {} not supported'.format(filetype))
            exit(1)

        return observations

    def import_feature(self, feature_type, feature_name, features):
        ''' Import an externally-computed feature for the given points3D.

            Attributes:
                feature_type (string)   :   type of the feature (point3D, point2D or camera feature)
                feature_name (string)   :   name of the feature
                features ({int -> _})   :   values of the features. {id -> feature_value}
        '''
        if len(features) == 0:
            logger.warning('Empty feature dictionary is passed')
            return      
   
        self.__features[feature_type].append(feature_name)
     
        for id, value in features.items():
            if feature_type == 'point3D':
                self.__points3D[id].set_feature(feature_name, value)
            else:
                logger.critical('Only features for points3D are supported now')
                exit(1)
        logger.info('Imported {} feature: {} for {} elements'.format(feature_type, feature_name, len(features)))


    ''' ************************************************ Getters ************************************************ '''
    def get_point2D(self, p2D_id):
        ''' Return the point2D with the given id.

            Attributes:
                p2D_id (int)    :   id of the point2D
        '''
        if p2D_id in self.__points2D and self.__points2D[p2D_id]:
            return self.__points2D[p2D_id]
        else:
            logger.error('Point2D with id {} does not exist'.format(p2D_id))

    def get_point3D(self, p3D_id):
        ''' Return the point3D with the given id.

            Attributes:
                p3D_id (int)    :   id of the point3D
        '''
        if p3D_id in self.__points3D and self.__points3D[p3D_id]:
            return self.__points3D[p3D_id]
        else:
            logger.error('Point3D with id {} does not exist'.format(p3D_id))
    
    def get_number_of_points3D(self):
        ''' Return the number of points3D in the reconstruction
        '''
        return len(self.__points3D)

    def get_camera(self, cam_id):
        ''' Return the camera with the given id.

            Attributes:
                cam_id (int)    :   id of the camera
        '''
        if cam_id in self.__cameras and self.__cameras[cam_id]:
            return self.__cameras[cam_id]
        else:
            logger.error('Camera with id {} does not exist'.format(cam_id))

    def get_feature_names(self, feature_type):
        ''' Return the feature names of the features that have been computed or imported for given type.

            Attributes:
                feature_type (string)   :   get the features of this type
            
            Return:
                features [string]       :   features
        '''
        if feature_type in self.__features.keys():
            return self.__features[feature_type].copy()
        else:
            logger.critical('Type {} is not a valid feature type'.format(feature_type))
            exit(1)
    

    ''' ************************************************ Modifiers ************************************************ '''
    def remove_points3D(self, p3D_ids):
        ''' Remove the given points3D from the reconstruction.

            Attributes:
                p3D_id (list(int))    :   ids of the point3D to remove
        '''
        num_deleted_points = {'3D' : 0, '2D' : 0}
        for p3D_id in p3D_ids:
            if p3D_id not in self.__points3D:
                logger.warning('Attempt to delete a non existing point3D {}'.format(p3D_id))
                continue

            p3D = self.__points3D[p3D_id]
            
            cameras_seing_p3D = p3D.get_cameras_seing_me()
            for cam_id in cameras_seing_p3D:                                                # Remove p3D_id from the observing cameras
                self.__cameras[cam_id].remove_point3D(p3D_id)
                
            p3D_points2D = [p3D.get_observation(cam_id) for cam_id in cameras_seing_p3D]    # Delete the corresponding points2D
            for p2D_id in p3D_points2D:
                try:
                    del self.__points2D[p2D_id]
                    num_deleted_points['2D'] += 1
                except KeyError:
                    logger.warning('Attempt to delete a non existing point2D {}'.format(p2D_id))
            
            del self.__points3D[p3D_id]
            num_deleted_points['3D'] += 1
        
        logger.info('Deleted {} points3D and {} points2D'.format(num_deleted_points['3D'], num_deleted_points['2D']))

    def remove_point2D(self, p2D_ids):
        ''' Remove the given points2D from the reconstruction. Not yet supported.
            Missing index from p3D_id to cam_id (observations).
            The removal would be now very slow (checking the VALUES of p3D.observations() (p2D_id) instead
            of the KEYS(cam_id)) 

            Attributes:
                p2D_ids (list(int)) : ids of the point2D to remove
        '''
        logger.critical('Method remove_point2D is not yet supported')
        exit(1)

        '''for p2D_id in p2D_ids:
            p2D = self.__points2D[p2D_id]
            
            p2D_p3D = self.__points3D[p2D.get_point3D]
            p2D_p3D.remove(observation(cam_id))         # cam_id cannot be efficently retrieved   
        '''


    ''' ************************************************ Geometric ************************************************ '''
    def project_p3D(self, p3D_id, undistort=False):
        ''' Get the projections of a point3D in all the observing cameras

            Attributes:
                p3D_id (int)                                    :   id of the point3D
                undistort (bool)                                :   apply undistortion (optional)
            
            Return:
                projections_of_p3D ({int -> np.matrix(2,1)})     :  image coordinates of the point3D {cam_id -> projection} 
        '''
        if p3D_id not in self.__points3D:
            logger.critical('Point3D with id: {} does not exist'.format(p3D_id))
            exit(1)
                   
        cams_seing_p3D = self.__points3D[p3D_id].get_cameras_seing_me()     # Ids of the cameras seing the point3D
        p3D = self.__points3D[p3D_id]                                       # Point to project
        projections_of_p3D = {}                                             # Projections of the point

        for cam_id in cams_seing_p3D:
            cam_seing_p3D = self.__cameras[cam_id]                                                     

            R_cw = cam_seing_p3D.get_rotation_matrix(GeometrySettings.RotationMatrixType.BLOCK_EXCHANGE)           # Rotation of the camera in world coordinate system
            t_cw = cam_seing_p3D.get_translation_vector(GeometrySettings.TranslationVectorType.BLOCK_EXCHANGE)     # Camera center in world coordinate system
            K = cam_seing_p3D.get_camera_matrix_opencv()
            
            t_wc = (-R_cw).dot(t_cw)                        # Translation of the world coordinate system wrt the camera
            T = np.append(R_cw, t_wc, axis=1)               # 3x4 transformation matrix
           
            p3D_w = p3D.get_coordinates(homogeneous=True)   # Homogeneous P3D coordinates in the world coordinate system
            p3D_c = T.dot(p3D_w)                            # P3D coordinates in the camera coordinate system
            p3D_i = p3D_c / p3D_c[2]                        # Project and get the image coordinates (x, y, 1) of p3D_c

            if undistort:
                p3D_i = self.undistort_p3D(p3D_i, cam_id)   # Correct distortion

            p3D_ik = K.dot(p3D_i)                           # Apply camera matrix
            
            logger.debug('Projection of P3D {} on cam {}: {}'.format(p3D_id, cam_id, p3D_ik.tolist()))
            projections_of_p3D[cam_id] = p3D_ik

        return projections_of_p3D

    def undistort_p3D(self, p3D_xy, cam_id):
        ''' Apply radial and tangential correction to the coordinates of the point3D.

            Attributes:
                p3D_xy (np.matrix)      :   image (distorted) coordinates of the point3D. Homogeneous vector 3 x 1 
                cam_id (int)            :   id of the camera seing p3D         

            Return:
                p3D_iu (np.matrix)  :   undistorted coordinates
        '''
        if cam_id not in self.__cameras:
            logger.critical('Camera with id {} does not exist'.format(cam_id))
            exit(1)
        
        p3D_xy_und = np.matrix([[0.0], [0.0], [1.0]])                                   # Undistorted image coordinates
        
        k1, k2, k3 = self.__cameras[cam_id].get_radial_distortion_params()              # Radial distortion parameters
        p1, p2 = self.__cameras[cam_id].get_tangential_distortion_params()              # Tangential distortion parameters
        r2 = pow(p3D_xy[0], 2) + pow(p3D_xy[1], 2)                                      # R2 factor

        p3D_xy_und[0] = (p3D_xy[0] * (1 + (k1 * r2) + (k2 * pow(r2,2)) + (k3 * pow(r2, 3)))) + ((2 * p1 * p3D_xy[0] * p3D_xy[1]) + (p2 * (r2 + (2 * pow(p3D_xy[0], 2)))))
        p3D_xy_und[1] = (p3D_xy[1] * (1 + (k1 * r2) + (k2 * pow(r2,2)) + (k3 * pow(r2, 3)))) + ((p1 * (r2 + (2 * pow(p3D_xy[1], 2)))) + (2 * p2 * p3D_xy[0] * p3D_xy[1]))
        logger.debug('{} undistorted to {}'.format(p3D_xy.tolist(), p3D_xy_und.tolist()))
        
        return p3D_xy_und


    ''' ************************************************ Features ************************************************ '''
    def compute_mean_reprojection_errors(self):
        ''' Compute the mean reprojection error of all the points3D
        '''
        if len(self.__points3D) == 0:
            logger.critical('No points3D, no reprojection errors')
            exit(1)
        
        feature_name = 'mean_reprojection_error'
        self.__features['point3D'].append(feature_name)

        for p3D_id, p3D in self.__points3D.items():
            p3D_projections = self.project_p3D(p3D_id, undistort=True)
            p3D_mean_reprojection_error = 0
            
            for cam_id, p3D_reprojection in p3D_projections.items():
                p3D_reprojection = p3D_reprojection[0:2]                                    # Remove the homogeneous coordinate

                width, height = self.__cameras[cam_id].get_image_resolution()               # Convert 2D coordinates in the .out format (wrt image center)
                p3D_reprojection[0] = p3D_reprojection[0] - (width / 2)
                p3D_reprojection[1] = (height / 2) - p3D_reprojection[1]

                original_p2D_id = p3D.get_observation(cam_id)
                original_p2D_xy = self.__points2D[original_p2D_id].get_coordinates()
                err = np.linalg.norm(original_p2D_xy - p3D_reprojection)                    # Compute the reprojection error as the norm of the difference between the original and the projected 2D coordinates
                
                p3D_mean_reprojection_error += err  
                logger.debug('Reprojection error of P3D {} on cam {}: {}'.format(p3D_id, cam_id, err))
            
            p3D_mean_reprojection_error = p3D_mean_reprojection_error / len(p3D_projections)
            self.__points3D[p3D_id].set_feature(feature_name, p3D_mean_reprojection_error)
            logger.debug('Mean reprojection error of P3D {} is {}'.format(p3D_id, p3D_mean_reprojection_error))
        
        logger.info('Reprojection errors computed')

    def compute_max_intersection_angles(self, in_degree=False):
        ''' Compute the maximum intersection angle of all the points3D.

            Attributes:
                in_degree (bool)   :   save angles in degrees instead of radiants
        '''
        if len(self.__points3D) == 0:
            logger.critical('No points3D, no maximum intersection angles')
            exit(1)
        
        feature_name = 'max_intersec_angle'
        self.__features['point3D'].append(feature_name)
    
        for p3D_id, p3D in self.__points3D.items():
            cams_seing_p3D = p3D.get_cameras_seing_me()

            p3D_cams_distances = []
            for cam_id in cams_seing_p3D:           # Compute distance vector between p3D and the cameras seing it
                camera = self.__cameras[cam_id]
                camera_tranlsation = camera.get_translation_vector(GeometrySettings.TranslationVectorType.BLOCK_EXCHANGE)
                p3D_cams_distances.append(p3D.get_coordinates() - camera_tranlsation)
   
            arcoss_angles = []
            for dist_1_index in range(len(p3D_cams_distances)):
                for dist_2_index in range(dist_1_index + 1, len(p3D_cams_distances)):
                    dist_1 = p3D_cams_distances[dist_1_index]
                    dist_2 = p3D_cams_distances[dist_2_index]
                    dist_1_unitized = dist_1 / np.linalg.norm(dist_1)
                    dist_2_unitized = dist_2 / np.linalg.norm(dist_2)

                    dist_1_dist_2_cos_angle = (np.dot(dist_1_unitized.T, dist_2_unitized)) / (np.linalg.norm(dist_1_unitized)) * (np.linalg.norm(dist_1_unitized))
                    dist_1_dist_2_arcoss_angle = np.arccos(np.asscalar(dist_1_dist_2_cos_angle))
                    if in_degree:
                        dist_1_dist_2_arcoss_angle = np.degrees(dist_1_dist_2_arcoss_angle)
                    arcoss_angles.append(dist_1_dist_2_arcoss_angle)
            
            max_angle = max(arcoss_angles)
            self.__points3D[p3D_id].set_feature(feature_name, max_angle)
            logger.debug('Max intersection angle for p3D {} is {}'.format(p3D_id, max_angle)) 

        logger.info('Maximum intersection angles computed')

    def compute_multiplicities (self):
        ''' Compute the multiplicity (number of cameras seing each point3D).
        '''
        if len(self.__points3D) == 0:
            logger.critical('No points3D, no multiplicities')
            exit(1)
        
        feature_name = 'multiplicity'
        self.__features['point3D'].append(feature_name)
        
        for _, p3D in self.__points3D.items():
            p3D.set_feature(feature_name, len(p3D.get_cameras_seing_me()))
        
        logger.info('Multiplicities computed')


    ''' ************************************************ Statistic ************************************************ '''
    def compute_feature_statistics(self, features, ids_to_exclude=None):
        ''' Compute feature statistics of all the points3D or of a specif set (when ids_to_exclude is specified)

            Attributes:
                features [string]               :   features to consider
                ids_to_exclude ([int])          :   ids of points3D to exclude
            
            Return:
                feature_statistics ({string ->  
                    {string -> float}})         :   min, max, mean, median and std of each statistics 
                                                    {'feature' -> {'stat' -> float}} where 'stat' is 
                                                    ['min', 'max', 'mean', 'median', 'std', '5th', '95th']          
        '''
        feature_statistics = dict.fromkeys(features)
        for feature in features:
            feature_statistics[feature] = {}

        if ids_to_exclude:
            p3Ds_to_consider = [p3D for p3D_id, p3D in self.__points3D.items() if p3D_id not in ids_to_exclude]
        else:
            p3Ds_to_consider = [p3D for _, p3D in self.__points3D.items()]

        for feature in features:
            feature_statistics[feature]['min'] = np.min([p3D.get_feature(feature) for p3D in p3Ds_to_consider])
            feature_statistics[feature]['mean'] = np.mean([p3D.get_feature(feature) for p3D in p3Ds_to_consider])
            feature_statistics[feature]['std'] = np.std([p3D.get_feature(feature) for p3D in p3Ds_to_consider])
            feature_statistics[feature]['median'] = np.median([p3D.get_feature(feature) for p3D in p3Ds_to_consider])
            feature_statistics[feature]['max'] = np.max([p3D.get_feature(feature) for p3D in p3Ds_to_consider])
            feature_statistics[feature]['5th'] = np.percentile([p3D.get_feature(feature) for p3D in p3Ds_to_consider], 5)
            feature_statistics[feature]['95th'] = np.percentile([p3D.get_feature(feature) for p3D in p3Ds_to_consider], 95)

        return feature_statistics


    ''' ************************************************ Export ************************************************ '''
    def export_points3D_xyz_and_features(self, folder):
        ''' Export points3D coordinates and features in a .txt file. 
            Format <p3D_id> <x> <y> <z> <reprojection_error> <multiplicity> <max_intersection_angle>

            Attributes:
                folder (string) :   path to the output folder where "features.txt" will be created
        ''' 
        if not os.path.exists(folder):
           logger.error('Impossible to export "features.txt". Folder "{}" does not exist'.format(folder))
           return

        try:
            features_names = self.get_feature_names('point3D')
            with open(os.path.join(folder, 'features.txt'), 'w') as f_out:
                f_out.write('#id x y z {}\n'.format(' '.join(features_names)))

                for p3D_id, p3D in self.__points3D.items():
                    f_out.write('{} {}'.format(p3D_id, p3D.get_coordinates_as_string()))
                    for feature_name in features_names:
                        f_out.write(' {}'.format(p3D.get_feature(feature_name)))
                    f_out.write('\n')
        except IOError:
            logger.error('Cannot create file {}"'.format(os.path.join(folder, 'features.txt')))
            exit(1)

    def export_reconstruction(self, folder, format, filename=None):
        ''' Export the points2D, points2D and cameras in the given format.

            Attributes:
                folder (string)                     :   path to the output folder
                format (SupportedOutputFileFormat)  :   format of the output file  
                filename (string)                   :   name of the file (optional)
        '''
        if format == GeometrySettings.SupportedOutputFileFormat.OUT:
            return self.__export_out(folder, filename)
        elif format == GeometrySettings.SupportedOutputFileFormat.NVM:
            return self.__export_nvm(folder, filename)
        else:
            logger.critical('Invalid output format. Supported values: out and nvm')
            exit(1)

    def __export_out(self, folder, filename):
        ''' Export reconstruction in Bundler out format.

            Attributes:
                folder (string)     :   path to the output folder
                filename (string)   :   name of the file (optional)
            
            Return:
                out_path (string)   :   filepath to the generated .out
        ''' 
        if not filename:
            filename = time.strftime('%Y%m%d-%H%M%S.out')
        else:
            if '.out' not in filename:
                filename = filename + '.out'

        with open(os.path.join(folder, filename), 'w') as f_out:
            f_out.write('# Bundle file v0.3\n')          
            f_out.write('{} {}\n'.format(len(self.__cameras), len(self.__points3D)))

            for _, cam in self.__cameras.items():
                f, (k1, k2, _) = cam.get_focal(), cam.get_radial_distortion_params()
                f_out.write('{} {} {}\n'.format(f, k1, k2))

                R = [str(np.asscalar(r)) for r in np.nditer(cam.get_rotation_matrix(GeometrySettings.RotationMatrixType.BUNDLER_OUT))]        # Convert np.matrix to list of strings
                for r_index, r in enumerate(R):
                    f_out.write('{}\n'.format(r)) if (r_index+1) % 3 == 0 else f_out.write('{} '.format(r))
                
                t = [str(np.asscalar(r)) for r in np.nditer(cam.get_translation_vector(GeometrySettings.TranslationVectorType.BUNDLER_OUT))]  # Convert np.matrix to list of strings
                f_out.write('{}\n'.format(' '.join(t)))
        
            for _, p3D in self.__points3D.items():
                xyz = [str(np.asscalar(v)) for v in np.nditer(p3D.get_coordinates())]          
                f_out.write('{}\n'.format(' '.join(xyz)))

                rgb = [str(np.asscalar(v)) for v in np.nditer(p3D.get_color())]         
                f_out.write('{}\n'.format(' '.join(rgb)))   

                cam_seing_p3D = p3D.get_cameras_seing_me()
                f_out.write('{} '.format(len(cam_seing_p3D)))
                for cam_id in cam_seing_p3D:
                    p2D = self.__points2D[p3D.get_observation(cam_id)]
                    cam = self.__cameras[cam_id]
                    xy = [str(np.asscalar(v)) for v in np.nditer(p2D.get_coordinates())] 
                    f_out.write('{} {} {} '.format(cam_id, p2D.get_keypoint_index()[0], ' '.join(xy)))  

        return os.path.join(folder, filename)    

    def __export_nvm(self, folder, filename):
        logger.critical('Method not yet supported')
        exit(1)