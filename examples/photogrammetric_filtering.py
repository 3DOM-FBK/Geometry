from sys import path
from argparse import ArgumentParser

# Path to geometry library
path.insert(0, '../')

from geometry import Geometry
from geometrySettings import GeometrySettings

def main():
    parser = ArgumentParser(description='Photogrammetric filtering of sparse reconstructions')
    parser.add_argument('--input', help='Path to the input file [.out/.nvm]', required=True)
    parser.add_argument('--output', help='Path to the output folder', required=True)
    parser.add_argument('--intrinsics', help='Path to the file containing the full instrisic values of the cameras')
    parser.add_argument('--intrinsic_format', help='Format of the instrisic file')
    parser.add_argument('--debug', help='Run in debug mode', type=int, default=0)
    args = parser.parse_args()

if __name__ == '__main__':
    main()