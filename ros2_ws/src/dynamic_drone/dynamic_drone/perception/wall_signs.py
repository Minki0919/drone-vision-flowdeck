"""Image-derived wall sign quadrilateral, metric pose and arrow instruction."""
import cv2
import numpy as np

TURN={'up':0.,'left':np.pi/2,'right':-np.pi/2,'down':np.pi}

def arrow_image(label,size=40):
    image=np.full((size,size),255,np.uint8)
    # Canonical up sign, rotated without interpolating black/white edges.
    points=np.array([[.5,.12],[.83,.45],[.60,.45],[.60,.87],[.40,.87],[.40,.45],[.17,.45]])
    cv2.fillPoly(image,[(points*(size-1)).astype(np.int32)],0)
    return {'up':image,'right':np.rot90(image,3),'down':np.rot90(image,2),'left':np.rot90(image,1)}[label].copy()

def quadrilaterals(gray):
    result=[]
    for threshold in (8,12,20,35,50,70,90,110,140,180,210):
        mask=cv2.threshold(gray,threshold,255,cv2.THRESH_BINARY)[1]
        contours,_=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area=cv2.contourArea(contour)
            if not 250<area<gray.size*.85:continue
            hull=cv2.convexHull(contour)
            poly=cv2.approxPolyDP(hull,.025*cv2.arcLength(hull,True),True)
            if len(poly)!=4 or not cv2.isContourConvex(poly):continue
            q=poly.reshape(4,2).astype(np.float32)
            if (q[:,0]<3).any() or (q[:,1]<3).any() or (q[:,0]>gray.shape[1]-4).any() or (q[:,1]>gray.shape[0]-4).any():continue
            # TL, TR, BR, BL image order, independent of detector class.
            sums=q.sum(1);diff=q[:,1]-q[:,0]
            q=np.array([q[sums.argmin()],q[diff.argmin()],q[sums.argmax()],q[diff.argmax()]])
            lengths=np.linalg.norm(q-np.roll(q,1,axis=0),axis=1)
            if lengths.min()<12 or lengths.max()/lengths.min()>3:continue
            if any(np.linalg.norm(q.mean(0)-old.mean(0))<8 for old in result):continue
            result.append(q)
    return sorted(result,key=lambda q:cv2.contourArea(q),reverse=True)[:16]

def board_pose(corners,matrix,size):
    h=size/2;obj=np.array([[-h,h,0],[h,h,0],[h,-h,0],[-h,-h,0]],np.float32)
    K=np.asarray(matrix,np.float64).reshape(3,3)
    ok,r,t=cv2.solvePnP(obj,corners,K,np.zeros(5),flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or not np.isfinite(t).all() or not .3<t[2,0]<8:return None
    projected,_=cv2.projectPoints(obj,r,t,K,np.zeros(5));error=float(np.linalg.norm(projected.reshape(4,2)-corners,axis=1).mean())
    if error>2:return None
    x,y,z=t.ravel()
    return dict(relative_body=[float(z),float(-x),float(-y)],rotation_vector=r.ravel().tolist(),reprojection_error_px=error,corners=corners.tolist())


def arrow_shape_direction(gray):
    """Read the arrow tip from pixels in a rectified board, independent of YOLO."""
    h,w=gray.shape;pad=max(4,int(min(h,w)*.09));roi=gray[pad:h-pad,pad:w-pad]
    if roi.size==0 or float(np.std(roi))<5:return None
    binary=cv2.threshold(roi,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]
    contours,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    candidates=[]
    for contour in contours:
        area=cv2.contourArea(contour)
        if not roi.size*.04<area<roi.size*.65:continue
        x,y,bw,bh=cv2.boundingRect(contour)
        if min(x,y)<2 or x+bw>=roi.shape[1]-2 or y+bh>=roi.shape[0]-2:continue
        poly=cv2.approxPolyDP(contour,.025*cv2.arcLength(contour,True),True).reshape(-1,2).astype(float)
        if not 6<=len(poly)<=9:continue
        for i,tip in enumerate(poly):
            first=poly[i-1]-tip;second=poly[(i+1)%len(poly)]-tip
            # The head tip meets two diagonal edges; stem corners do not.
            def diagonal(vector):
                normal=vector/np.array([bw,bh]);return np.abs(normal).min()>.12 and np.abs(normal).min()/np.abs(normal).max()>.35
            if not diagonal(first) or not diagonal(second):continue
            vector=tip-(poly[i-1]+poly[(i+1)%len(poly)])/2
            if np.linalg.norm(vector)<min(bw,bh)*.18:continue
            vectors={'up':np.array([0,-1]),'down':np.array([0,1]),'left':np.array([-1,0]),'right':np.array([1,0])}
            unit=vector/np.linalg.norm(vector);label=max(vectors,key=lambda label:float(unit@vectors[label]));alignment=float(unit@vectors[label])
            if alignment>.90:candidates.append(dict(label=label,alignment=alignment,area_px=float(area),tip_pixel=(tip+[pad,pad]).tolist()))
    labels={c['label'] for c in candidates}
    return max(candidates,key=lambda c:c['area_px']) if len(labels)==1 else None
