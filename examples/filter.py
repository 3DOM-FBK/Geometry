import os
import sys
import json
import logging
import argparse
import numpy as np

# Path to geometry library
sys.path.insert(0, '../')

from geometry import Geometry
from geometrySettings import GeometrySettings

# Global variables
filter_threshold = 0                                                        # Filter treshold
equation_id = None                                                          # Id of the filtering equation to use
weight_id = None                                                            # Id of the weight factor to use
equations_ids = {'0' : lambda re, mu, ma, s: re + (1-mu) + (1-ma) + s,      # Filtering equations
            '1' : lambda re, ma, s: re + (1-ma) + s,
            '2' : lambda re : re,
            '3' : lambda ma : 1 - ma,
            '4' : lambda mu : 1 - mu,
            '5' : lambda s : s
            }
weight_ids = {'0' : lambda x: 1,                                   # Weight factors
            '1' : lambda mu, mu_max : 1 - (mu / mu_max),
            '2' : lambda mu, mu_95 : 1 - (mu / mu_95)
            }


''' ************************************************ Import ************************************************ '''
def import_sigma_features(path, geometry):
    ''' Import in geometry the sigma feature exported by Metashape. File format: x y z std_x std_y std_z std.
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

def import_photo_features(path, geometry):
    ''' Import reprojection error, multiplicity and max intersection angle computed by "compute_features.py".
        File format: id x y z mean_reprojection_error multiplicity max_intersec_angle
        Assume point order: same as in the .out/.nvm file

        Attributes:
            path (string)           :   path to the file generated with "compute_features.py"
            geometry (Geometry)     :   reference to the geometry instance
    '''
    if not os.path.isfile(path):
        logging.critical('File "{}" does not exist'.format(path))
        exit(1)
    
    repr_errs = {}                                                      # Reprojection errors
    mults = {}                                                          # Multiplicities
    max_angles = {}                                                     # Max intersection angles
    
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
        
        for p3D_id, line in enumerate(lines[1:]):
            tokens = line.replace('\t', ' ').strip().split(' ')
            if len(tokens) < 7:
                logging.critical('Invalid file format. Expected: <id> <x> <y> <z> '\
                    '<mean_reprojection_error> <multiplicity> <max_intersec_angle>')
                exit(1)
            repr_errs[p3D_id] = float(tokens[4])
            mults[p3D_id] = float(tokens[5])
            max_angles[p3D_id] = float(tokens[6])                       
    except IOError:
        logging.critical('File "{}" is not readable'.format(path))
        exit(1)
    
    geometry.import_feature('point3D', 'mean_reprojection_error', repr_errs)
    geometry.import_feature('point3D', 'multiplicity', mults)
    geometry.import_feature('point3D', 'max_intersec_angle', max_angles)


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

def compute_filter_threshold(geometry):
    # Compute stats of the features ('min', 'max', 'mean', 'median', 'std', '5th', '95th')
    p3Ds_feature_names = geometry.get_feature_names('point3D')
    p3Ds_feature_stats = geometry.compute_feature_statistics(p3Ds_feature_names)
    logging.info("Feature statistics: {}".format(json.dumps(p3Ds_feature_stats, indent=4, sort_keys=True)))

    accum = 0.0
    for f_name in p3Ds_feature_names:
        if f_name == 'max_intersec_angle':
             accum += 1 - logistic_normalisation(p3Ds_feature_stats[f_name]['median'], 
                                        p3Ds_feature_stats[f_name]['mean'], 
                                        p3Ds_feature_stats[f_name]['std']   
            )
        else:
            accum += logistic_normalisation(p3Ds_feature_stats[f_name]['median'], 
                                            p3Ds_feature_stats[f_name]['mean'], 
                                            p3Ds_feature_stats[f_name]['std']   
            )
    
    return accum


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
    if equation_id == '0':
        re_norm = logistic_normalisation(p3D.get_feature('mean_reprojection_error'), feature_stats['mean_reprojection_error']['mean'], feature_stats['mean_reprojection_error']['std'])
        mu_norm = logistic_normalisation(p3D.get_feature('multiplicity'), feature_stats['multiplicity']['mean'], feature_stats['multiplicity']['std'])
        ma_norm = logistic_normalisation(p3D.get_feature('max_intersec_angle'), feature_stats['max_intersec_angle']['mean'], feature_stats['max_intersec_angle']['std'])
        s_norm = logistic_normalisation(p3D.get_feature('sigma'), feature_stats['sigma']['mean'], feature_stats['sigma']['std'])
        score = equations_ids[equation_id](re_norm, mu_norm, ma_norm, s_norm)
    elif equation_id == '1':
        re_norm = logistic_normalisation(p3D.get_feature('mean_reprojection_error'), feature_stats['mean_reprojection_error']['mean'], feature_stats['mean_reprojection_error']['std'])
        ma_norm = logistic_normalisation(p3D.get_feature('max_intersec_angle'), feature_stats['max_intersec_angle']['mean'], feature_stats['max_intersec_angle']['std'])
        s_norm = logistic_normalisation(p3D.get_feature('sigma'), feature_stats['sigma']['mean'], feature_stats['sigma']['std'])
        score = equations_ids[equation_id](re_norm, mu_norm, ma_norm, s_norm)
    elif equation_id == '2':
        re_norm = logistic_normalisation(p3D.get_feature('mean_reprojection_error'), feature_stats['mean_reprojection_error']['mean'], feature_stats['mean_reprojection_error']['std'])
        score = equations_ids[equation_id](re_norm)
    elif equation_id == '3':
        ma_norm = logistic_normalisation(p3D.get_feature('max_intersec_angle'), feature_stats['max_intersec_angle']['mean'], feature_stats['max_intersec_angle']['std'])
        score = equations_ids[equation_id](ma_norm)
    elif equation_id == '4':
        mu_norm = logistic_normalisation(p3D.get_feature('multiplicity'), feature_stats['multiplicity']['mean'], feature_stats['multiplicity']['std'])
        score = equations_ids[equation_id](mu_norm)
    elif equation_id == '5':
        s_norm = logistic_normalisation(p3D.get_feature('sigma'), feature_stats['sigma']['mean'], feature_stats['sigma']['std'])
        score = equations_ids[equation_id](s_norm)
    else:
        logging.critical('Unknown equation id {}'.format(equation_id))
        exit(1)
    
    final_score = 0
    if weight_id == '0':
        final_score = weight_ids[weight_id](1) * score
    elif weight_id == '1':
        final_score = weight_ids[weight_id](p3D.get_feature('multiplicity'), feature_stats['multiplicity']['max']) * score
    elif weight_id == '2':
        final_score = weight_ids[weight_id](p3D.get_feature('multiplicity'), feature_stats['multiplicity']['95th']) * score
    else:
        logging.critical('Unknown weight factor id {}'.format(weight_id))
        exit(1)

    logging.debug('P3D: {} score: {}'.format(p3D.get_id(), final_score))
    return final_score

def filter_points3D(geometry):
    ''' Apply photogrammetric filtering to the points3D (and corresponding point2D) of the given reconstruction.

        Attributes:
            geometry (Geometry)     :   reference to the geometry instance
    '''
    p3Ds_feature_names = geometry.get_feature_names('point3D')
    p3Ds_feature_stats = geometry.compute_feature_statistics(p3Ds_feature_names)

    p3Ds_to_delete = []
    for p3D_id in range(0, geometry.get_number_of_points3D()):
        p3D = geometry.get_point3D(p3D_id)
        p3D_score = compute_point3D_score(p3D, p3Ds_feature_stats)
        if p3D_score > filter_threshold:
            p3Ds_to_delete.append(p3D_id)
            logging.debug('Point3D {} deleted. Score: {}'.format(p3D_id, p3D_score))

    geometry.remove_points3D(p3Ds_to_delete)
    logging.info('Deleted {} points3D'.format(len(p3Ds_to_delete)))


def main():
    parser = argparse.ArgumentParser(description='Photogrammetric filtering of sparse reconstructions.')
    parser.add_argument('--input', help='Path to the input file [.out/.nvm]', required=True)
    parser.add_argument('--output', help='Path to the output folder', required=True)
    parser.add_argument('--intrinsics', help='Path to the file containing the full instrisic values of the cameras', required=True)                                         
    parser.add_argument('--intrinsic_format', help='Format of the instrisic file', required=True)                                                                           
    parser.add_argument('--features', help='Path to the file containing the photogrammetric features', required=True)
    parser.add_argument('--sigma', help='Path to the file containing the sigma features', required=True)
    parser.add_argument('--eq', help='Filtering equation id', required=True)
    parser.add_argument('--weight', help='Weight factor id', required=True)
    parser.add_argument('--debug', help='Run in debug mode', type=int, default=0)
    args = parser.parse_args()

    if args.debug == 1:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    logging.basicConfig(format='%(levelname)-6s %(asctime)s:%(msecs)d [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S', filename=os.path.join(args.output, "log_filter.txt"), filemode='w', level=log_level)
    
    # Get id of the filter weights and equations
    global equation_id, weight_id
    equation_id, weight_id = args.eq, args.weight
    logging.info('Filtering params: equation id: {}, weight id: {}'.format(equation_id, weight_id))

    geometry = Geometry()

    # Load reconstruction
    geometry.load_reconstruction(args.input)

    # Load camera intrinsics (per se not needed, but import them to avoid errors when generating the filtered .out)
    if args.intrinsic_format.lower() == 'metashape':
        geometry.load_full_camera_intrinsics(args.intrinsics, GeometrySettings.InstriscsFormatType.METASHAPE)
    elif args.intrinsic_format.lower() == 'opencv':
        geometry.load_full_camera_intrinsics(args.intrinsics, GeometrySettings.InstriscsFormatType.OPENCV)
    else:
        logging.critical('Unknown intrinsic format. Supported values: [\'opencv\', \'metashape\']')
        exit(1)

    # Import photogrammetric and sigma features
    import_photo_features(args.features, geometry)
    import_sigma_features(args.sigma, geometry)

    # Compute filter theshold
    global filter_threshold
    filter_threshold = compute_filter_threshold(geometry)
    logging.info("Computed filter threshold: {}".format(filter_threshold))

    # Filter the points
    filter_points3D(geometry)
    
    # Export filtered dataset in .out format
    out_path = geometry.export_reconstruction(args.output, GeometrySettings.SupportedOutputFileFormat.OUT)
    logging.info('Filtered reconstruction saved in {}'.format(os.path.join(out_path)))


if __name__ == '__main__':
    main()