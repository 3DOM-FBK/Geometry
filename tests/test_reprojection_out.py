import logging
import numpy as np
from argparse import ArgumentParser

from geometry import Geometry
from geometrySettings import GeometrySettings


def main():
    parser = ArgumentParser(description='Test the .out reprojection without using Block Exchange format')
    parser.add_argument('--input', help='Path to the input file [.out/.nvm]', required=True)
    parser.add_argument('--intrinsics', help='Path to the file containing the full instrisic values of the cameras')
    parser.add_argument('--intrinsic_format', help='Format of theinstrisic file')
    args = parser.parse_args()

    logging.basicConfig(format='%(levelname)-6s %(asctime)s:%(msecs)d [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S', filename="log.txt", filemode='w', level=logging.DEBUG)
    
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

    # Test it with the point3D with id 0
    p3D_0 = geometry.get_point3D(0)
    print('Point3D: {}'.format(p3D_0))
    for cam_id in p3D_0.get_cameras_seing_me():
        camera_seing_p3D_0 = geometry.get_camera(cam_id)                            
        # p2D_projection = geometry.get_point2D(p3D_0.get_observation(cam_id))
        # print('True p2D:\n{}'.format(p2D_projection.get_coordinates()))
    
        #R = camera_seing_p3D_0.get_rotation_matrix(GeometrySettings.RotationMatrixType.BUNDLER_OUT)
        #t = camera_seing_p3D_0.get_translation_vector(GeometrySettings.TranslationVectorType.BUNDLER_OUT)
        #K = camera_seing_p3D_0.get_camera_matrix_opencv()

        print(camera_seing_p3D_0.get_translation_vector(GeometrySettings.TranslationVectorType.BLOCK_EXCHANGE))

        '''T = np.append(R, t, axis=1)                                     # 3x4 transformation matrix

        p3D_0_w = p3D_0.get_coordinates(homogeneous=True)
        p3D_0_c = T.dot(p3D_0_w)                                        # P3D coordinates in the camera coordinate system
        p3D_0_i = (-p3D_0_c) / p3D_0_c[2]                               # Project and get the image coordinates (x, y, 1) of p3D_c

        p3D_0_i = geometry.undistort_p3D(p3D_0_i, cam_id)               # Correct distortion
        p3D_0_ik = K.dot(p3D_0_i)                                       # Apply camera matrix

        width, height = camera_seing_p3D_0.get_image_resolution()
        p3D_0_ik[0] = p3D_0_ik[0] - (width / 2)
        p3D_0_ik[1] = p3D_0_ik[1] - (height / 2)
        print('Estimated projection:\n{}'.format(p3D_0_ik)) '''
        

if __name__ == '__main__':
    main()