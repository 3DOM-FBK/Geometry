import unittest

from geometry import Geometry

class TestProjection(unittest.TestCase):
    ''' Test the projection 
    '''
    def setUp(self):
        self.input = '../data/modena_only_ter/soloterrestri.out'
        self.intrinsics = '../data/modena_only_ter/cal_test_terrestri.txt'
        self.intrinsics_format = 'metashape'
    
    def test_(self):
        self.assertEqual(self.s.upper(), 'FOO')

if __name__ == '__main__':
    unittest.main()