import logging
from argparse import ArgumentParser

from geometry import Geometry
from geometrySettings import GeometrySettings

def main():
    parser = ArgumentParser(description='Geometry: a tool for managing sparse reconstrunctions')
    parser.add_argument('--input', help='Path to the input file [.out/.nvm]', required=True)
    parser.add_argument('--output', help='Path to the output folder', required=True)
    parser.add_argument('--intrinsics', help='Path to the file containing the full instrisic values of the cameras')
    parser.add_argument('--intrinsic_format', help='Format of theinstrisic file')
    parser.add_argument('--debug', help='Run in debug mode', type=int, default=0)
    args = parser.parse_args()

    if args.debug == 1:                                                                                                         # Set logging
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    logging.basicConfig(format='%(levelname)-6s %(asctime)s:%(msecs)d [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S', filename="log.txt", filemode='w', level=log_level)

    geometry = Geometry()                                                                                                                                                                                 
    geometry.load_reconstruction(args.input)                                                                                                                                                            
   
    if args.intrinsics:                                                                                                         # Load full camera intrinsics                                                          
        if not args.intrinsic_format:
            logging.critical('No intrinsic format is specified. Supported values: [\'opencv\', \'metashape\']')
            exit(1)
        elif args.intrinsic_format == 'opencv':
            geometry.load_full_camera_intrinsics(args.intrinsics, GeometrySettings.InstriscsFormatType.OPENCV)                  
        elif args.intrinsic_format == 'metashape':
            geometry.load_full_camera_intrinsics(args.intrinsics, GeometrySettings.InstriscsFormatType.METASHAPE)
        else:
            logging.critical('Unknown intrinsic format. Supported values: [\'opencv\', \'metashape\']')
            exit(1)

    geometry.compute_mean_reprojection_errors()
    geometry.compute_max_intersection_angles(use_degree=True)

    geometry.export_points3D_coordinates_and_features(args.output)


if __name__ == '__main__':
    main()
    