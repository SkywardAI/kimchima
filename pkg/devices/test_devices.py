import unittest
import platform
import torch

from pkg.devices import Devices, get_device

class TestDevices(unittest.TestCase):

    def test_get_device(self):

        # Test if the device is a Mac silicon
        if platform.system() == 'Darwin':
            self.assertEqual(get_device(), Devices.Silicon)
        
        # Test if the device is a GPU
        if platform.system() != 'Darwin' and torch.cuda.is_available():
            self.assertEqual(get_device(), Devices.GPU)
        
        # Test if the device is a CPU
        if platform.system() != 'Darwin' and not torch.cuda.is_available():
            self.assertEqual(get_device(), Devices.CPU)

