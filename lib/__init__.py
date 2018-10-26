'''
Copyright (C) 2018  Sanchayan Santra

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

from lib import get_laplacian_4neigh, dehaze_patch, en_haze, discard_patch
from lib import compute_A_DCP, compute_A_Tang
from lib import get_patch_indices

from lib import patch_variance_test, patch_edge_test, patch_airlight_angle_test
