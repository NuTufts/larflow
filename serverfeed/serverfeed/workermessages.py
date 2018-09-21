import os,sys,re
from larcv import larcv

"""
This module contains the definitions for worker/broker messaging
"""

# Messages from worker
PPP_READY     = "\x01"
PPP_HEARTBEAT = "\x02"

# Decoding meta message (larcv1)
def decode_larcv1_metamsg( metamsg, override_plane=None ):
    """
    decode meta messages, which look like: 'Plane 65535 (rows,cols) = (0,0) ... Left Top (0,0) ... Right Bottom (0,0)\n'

    constructor:
    ImageMeta(const double width=0.,     const double height=0.,
      const size_t row_count=0., const size_t col_count=0,
      const double origin_x=0.,  const double origin_y=0.,
      const PlaneID_t plane=::larcv::kINVALID_PLANE)
    """
    meta_nums = [ int(x) for x in re.findall("\d+",metamsg.decode("ascii")) ]
    width    = meta_nums[5]-meta_nums[3]
    height   = meta_nums[4]-meta_nums[6]
    rows     = meta_nums[1]
    cols     = meta_nums[2]
    plane    = meta_nums[0]
    if override_plane is not None:
        plane = override_plane
    origin_x = meta_nums[3]
    origin_y = meta_nums[4]
    meta = larcv.ImageMeta( width, height, rows, cols, origin_x, origin_y, plane )

    return meta


