import os
import logging
from sys import path
from argparse import ArgumentParser

# Path to geometry library
path.insert(0, '../')

from geometry import Geometry
from geometrySettings import GeometrySettings

def main():
    parser = ArgumentParser(description='Create a file containing 3D points and photogrammetric features')
    parser.add_argument('--input', help='Path to the input file [.out/.nvm]', required=True)
    parser.add_argument('--output', help='Path to the output folder', required=True)
    parser.add_argument('--intrinsics', help='Path to the file containing the full instrisic values of the cameras', required=True)
    parser.add_argument('--intrinsic_format', help='Format of theinstrisic file', required=True)
    parser.add_argument('--debug', help='Run in debug mode', type=int, default=0)
    args = parser.parse_args()

    if args.debug == 1:                                                                                                         # Set logging
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    logging.basicConfig(format='%(levelname)-6s %(asctime)s:%(msecs)d [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S', filename=os.path.join(args.output," log.txt"), filemode='w', level=log_level)

    geometry = Geometry()                                                                                                                                                                                 
    
    # Load reconstruction
    geometry.load_reconstruction(args.input)                                                                                                                                                            
   
    # Load full camera intrinsics                                                                                                                                                         
    if args.intrinsic_format == 'opencv':
        geometry.load_full_camera_intrinsics(args.intrinsics, GeometrySettings.InstriscsFormatType.OPENCV)                  
    elif args.intrinsic_format == 'metashape':
        geometry.load_full_camera_intrinsics(args.intrinsics, GeometrySettings.InstriscsFormatType.METASHAPE)
    else:
        logging.critical('Unknown intrinsic format. Supported values: [\'opencv\', \'metashape\']')
        exit(1)

    # Compute photogrammetric features
    geometry.compute_mean_reprojection_errors()
    geometry.compute_multiplicities()
    geometry.compute_max_intersection_angles(in_degree=True)

    # Export features
    geometry.export_points3D_xyz_and_features(args.output)
    logging.info('Photogrammetric features exported in {}/features.txt'.format(os.path.join(args.output)))


if __name__ == '__main__':
    main()
    