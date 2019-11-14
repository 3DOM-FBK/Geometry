from enum import Enum

class GeometrySettings:
    ''' Geometry static settings and format types
    '''
    # Supported file formats
    class SupportedInputFileType(Enum):
        OUT = 0
        NVM = 1

    # Rotation matrix formats
    class RotationMatrixType(Enum):
        BUNDLER_OUT = 0
        BLOCK_EXCHANGE = 1

    # Translation vector formats
    class TranslationVectorType(Enum):
        BUNDLER_OUT = 0
        BLOCK_EXCHANGE = 1

    # Instrinsics format
    class InstriscsFormatType(Enum):
        OPENCV = 0
        METASHAPE = 1

    # Camera types
    class CameraType(Enum):
        GENERIC = 0
        TER = 1
        UAV = 2

    # Point types
    class PointType(Enum):
        GENERIC = 0
        TER = 1
        UAV = 2
        TER_UAV = 3