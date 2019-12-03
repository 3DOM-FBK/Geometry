import os
import sys
import logging
import argparse
import numpy as np

# Path to geometry library
sys.path.insert(0, '../')

from geometry import Geometry
from geometrySettings import GeometrySettings

# Global settings
weight_by_multiplicity = False
filtering_threshold = 0

''' ************************************************ Import ************************************************ '''
def import_sigma_features(path, geometry):
    ''' Import in geometry the gamma feature exported by Metashape. File format: x y z std_x std_y std_z std.
        First row is the header.

        Attributes:
            path (string)           :   path to the file containing the sigma features
            geometry (Geometry)     :   reference to the geometry instance
    '''
    if not os.path.isfile(path):
        logging.critical('File "{}" does not exist'.format(path))
        exit(1)
    
    sigma_features = {}
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
                        
            for p3D_id, line in enumerate(lines[1:]):
                tokens = line.replace('\t', ' ').strip().split(' ')
                if len(tokens) < 8:
                    logging.critical('Invalid instrinsic file format found at line {}. ' \
                        'Supported file format: <id> <x> <y> <z> <std_x> <std_y> <std_z> <std>'.format(line))
                    exit(1)

                # Use line number as id because the Metashape id is weird. 
                # ASSUMPTION: ordering in this file is equal to the ordering in the out/nvm file.
                sigma_features[p3D_id] = float(tokens[4])               
    except IOError:
        logging.critical('File "{}" is not readable'.format(path))
        exit(1)

    geometry.import_feature('point3D', 'sigma', sigma_features)


''' ************************************************ Statistics ************************************************ '''
def feature_scaling_normalisation(value, min, max):
        ''' Normalise a given value between 0 and 1 using the feature scaling method
        
        Args:
            value (float)   :   value to normalise.
            min (float)     :   minimum value in the data.
            max (float)     :   maximum value in the data.
        
        Return:
            result (float)  :   Normalised value 
        '''
        result = ((((value - min) * (1 - 0)) / float(max - min)))  
        return result

def logistic_normalisation(value, mean, std):
    ''' Normalise a value between 0 and 1 using a logistic function (Mauro version)

    Args:
        value (float)   :   value to normalise.
        mean (float)    :   mean of the data
        std (float)     :   standard deviation of the data 
    '''
    x = (2*(value - mean)) / std
    return 1 / (1 + np.exp(-x))


''' ************************************************ Filtering ************************************************ '''
def compute_point3D_score(p3D, feature_stats):
    ''' Computes the score of each point3D as a multiplicity-weighted (optional) summation of reprojection_error, maximum 
        intersection angle, multiplicity and sigma. 

        Attributes:
            p3D (Point3D)                                       :   target point3D   
            feature_stats ({string -> {string -> float}})       :   statistics of each point3D feature {feature -> {statistic -> value}}   
        
        Return:
            score (float)                                       :   score of the given point3D
    '''
    score = 0
    score += logistic_normalisation(p3D.get_feature('mean_reprojection_error'), feature_stats['mean_reprojection_error']['mean'], 
        feature_stats['mean_reprojection_error']['std'])
    score += logistic_normalisation(p3D.get_feature('sigma'), feature_stats['sigma']['mean'], feature_stats['sigma']['std'])
    score += 1 - logistic_normalisation(p3D.get_feature('multiplicity'), feature_stats['multiplicity']['mean'], 
        feature_stats['multiplicity']['std'])
    score += 1 - logistic_normalisation(p3D.get_feature('max_intersec_angle'), feature_stats['max_intersec_angle']['mean'], 
        feature_stats['max_intersec_angle']['std'])
    
    if weight_by_multiplicity:
        score = (p3D.get_feature('multiplicity') / feature_stats['multiplicity']['max']) * score
    
    logging.debug('P3D: {} score: {}'.format(p3D.get_id(), score))
    return score

def filter_points3D(geometry):
    ''' Apply photogrammetric filtering to the points3D (and corresponding point2D) of the given reconstruction.

        Attributes:
            geometry (Geometry)     :   reference to the geometry instance
    '''
    p3Ds_feature_names = geometry.get_feature_names('point3D')
    p3Ds_feature_stats = geometry.compute_feature_statistics(p3Ds_feature_names)
    print(p3Ds_feature_stats)

    p3Ds_to_delete = []
    for p3D_id in range(0, geometry.get_number_of_points3D()):
        p3D = geometry.get_point3D(p3D_id)
        p3D_score =  compute_point3D_score(p3D, p3Ds_feature_stats)
        if p3D_score > filtering_threshold:
            p3Ds_to_delete.append(p3D_id)
            logging.info('Point3D {} deleted. Score: {}'.format(p3D_id, p3D_score))

    geometry.remove_points3D(p3Ds_to_delete)
    logging.info('Filtering: deleted {} points3D'.format(len(p3Ds_to_delete)))


def main():
    parser = argparse.ArgumentParser(description='Photogrammetric filtering of sparse reconstructions')
    parser.add_argument('--input', help='Path to the input file [.out/.nvm]', required=True)
    parser.add_argument('--output', help='Path to the output folder', required=True)
    parser.add_argument('--intrinsics', help='Path to the file containing the full instrisic values of the cameras', required=True)
    parser.add_argument('--intrinsic_format', help='Format of the instrisic file', required=True)
    parser.add_argument('--sigma', help='Path to the sigma features file', required=True)
    parser.add_argument('--threshold', help='Filtering equation delete threshold. Points3D with a score higher than this will be deleted.', type=float, required=True)
    parser.add_argument('--weight_by_multiplicity', help='Use multiplicity weighting in the filtering equation', type=bool, default=False)
    parser.add_argument('--debug', help='Run in debug mode', type=int, default=0)
    args = parser.parse_args()

    if args.debug == 1:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    logging.basicConfig(format='%(levelname)-6s %(asctime)s:%(msecs)d [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S', filename=os.path.join(args.output," log.txt"), filemode='w', level=log_level)

    global weight_by_multiplicity, filtering_threshold
    weight_by_multiplicity = args.weight_by_multiplicity
    filtering_threshold = args.threshold
    logging.info('Filtering params: multiplicity as weigth: {}, filtering threshold: {}'.format(weight_by_multiplicity, filtering_threshold))

    geometry = Geometry()

    geometry.load_reconstruction(args.input)
    if args.intrinsic_format.lower() == 'metashape':
        geometry.load_full_camera_intrinsics(args.intrinsics, GeometrySettings.InstriscsFormatType.METASHAPE)
    elif args.intrinsic_format.lower() == 'opencv':
        geometry.load_full_camera_intrinsics(args.intrinsics, GeometrySettings.InstriscsFormatType.OPENCV)
    else:
        logging.critical('Unknown intrinsic format. Supported values: [\'opencv\', \'metashape\']')
        exit(1)

    geometry.compute_mean_reprojection_errors()
    geometry.compute_multiplicities()
    geometry.compute_max_intersection_angles(in_degree=True)
    import_sigma_features(args.sigma, geometry)

    geometry.export_points3D_xyz_and_features(args.output)

    filter_points3D(geometry)
    
    geometry.export_reconstruction(args.output, GeometrySettings.SupportedOutputFileFormat.OUT)


if __name__ == '__main__':
    main()