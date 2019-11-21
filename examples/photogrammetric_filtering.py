import os
import sys
import logging
import argparse

# Path to geometry library
sys.path.insert(0, '../')

from geometry import Geometry
from geometrySettings import GeometrySettings

def import_sigma_features(path, geometry, have_header=True):
    ''' Import in geometry the gamma feature. File format: x y z std_x std_y std_z std.
        First row is the header.

        Attributes:
            path (string)           :   path to the file containing the sigma features
            geometry (Geometry)     :   reference to the geometry instance
            have_header (bool)      :   specify if the file has a header
    '''
    if not os.path.isfile(path):
        logging.critical('File "{}" does not exist'.format(path))
        exit(1)
    
    sigma_features = {}
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
            
            if have_header:
                start_index = 1
            else:
                start_index - 0
            
            for line in lines[start_index:]:
                tokens = line.replace('\t', ' ').strip().split(' ')
                if len(tokens) < 8:
                    logging.critical('Invalid instrinsic file format found at line {}. ' \
                        'Supported file format: <id> <x> <y> <z> <std_x> <std_y> <std_z> <std>'.format(line))
                    exit(1)

                sigma_features[int(tokens[0])] = float(tokens[7])
    except IOError:
        logging.critical('File "{}" is not readable'.format(path))
        exit(1)

    geometry.import_feature('point3D', 'sigma', sigma_features)

def main():
    parser = argparse.ArgumentParser(description='Photogrammetric filtering of sparse reconstructions')
    parser.add_argument('--input', help='Path to the input file [.out/.nvm]', required=True)
    parser.add_argument('--output', help='Path to the output folder', required=True)
    parser.add_argument('--intrinsics', help='Path to the file containing the full instrisic values of the cameras', required=True)
    parser.add_argument('--intrinsic_format', help='Format of the instrisic file', required=True)
    parser.add_argument('--threshold', help='Filtering equation delete threshold', type=float, required=True)
    parser.add_argument('--weight_by_multiplicity', help='Use multiplicity weighting in the filtering equation', type=bool, default=False)
    parser.add_argument('--sigma', help='Path to the sigma features file', required=True)
    parser.add_argument('--debug', help='Run in debug mode', type=int, default=0)
    args = parser.parse_args()

    if args.debug == 1:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    logging.basicConfig(format='%(levelname)-6s %(asctime)s:%(msecs)d [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S', filename=os.path.join(args.output," log.txt"), filemode='w', level=log_level)

    geometry = Geometry()
    
    geometry.load_reconstruction(args.input)
    if args.intrinsic_format.lower() == 'metashape':
        geometry.load_full_camera_intrinsics(args.intrinsics, GeometrySettings.InstriscsFormatType.METASHAPE)
    elif args.intrinsic_format.lower() == 'opencv':
        geometry.load_full_camera_intrinsics(args.intrinsics, GeometrySettings.InstriscsFormatType.OPENCV)
    else:
        logging.critical('Unknown intrinsic format. Supported values: [\'opencv\', \'metashape\']')
        exit(1)
    
    import_sigma_features(args.sigma, geometry)
    geometry.export_points3D_xyz_and_features(args.output)


if __name__ == '__main__':
    main()