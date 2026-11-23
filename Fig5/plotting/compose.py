"""Frozen Matplotlib component composition."""

import numpy as np


def compose(arrays,composition,shape):
    out=np.full(shape,255,np.uint8)
    for row in composition:
        arr=arrays[row['part']]
        if row.get('crop_xy_pixels') is not None:
            x0,y0,x1,y1=row['crop_xy_pixels'];arr=arr[y0:y1,x0:x1]
        if row.get('crop_y_pixels') is not None:arr=arr[slice(*row['crop_y_pixels'])]
        x,y=row['offset_xy_pixels'];h,w=arr.shape[:2]
        l,t=max(0,-x),max(0,-y);r,b=min(w,shape[1]-x),min(h,shape[0]-y)
        assert l<r and t<b
        assert not np.any(arr[:t]!=255) and not np.any(arr[b:]!=255)
        assert not np.any(arr[:,:l]!=255) and not np.any(arr[:,r:]!=255)
        dst=out[y+t:y+b,x+l:x+r];dst[:]=np.minimum(dst,arr[t:b,l:r])
    return out
