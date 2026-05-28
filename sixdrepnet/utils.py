import os
import math
import urllib.request
from math import cos, sin

import numpy as np
import torch
#from torch.utils.serialization import load_lua
import scipy.io as sio
import cv2


def plot_pose_cube(img, yaw, pitch, roll, tdx=None, tdy=None, size=150.):
    # Input is a cv2 image
    # pose_params: (pitch, yaw, roll, tdx, tdy)
    # Where (tdx, tdy) is the translation of the face.
    # For pose we have [pitch yaw roll tdx tdy tdz scale_factor]

    p = pitch * np.pi / 180
    y = -(yaw * np.pi / 180)
    r = roll * np.pi / 180
    if tdx != None and tdy != None:
        face_x = tdx - 0.50 * size 
        face_y = tdy - 0.50 * size

    else:
        height, width = img.shape[:2]
        face_x = width / 2 - 0.5 * size
        face_y = height / 2 - 0.5 * size

    x1 = size * (cos(y) * cos(r)) + face_x
    y1 = size * (cos(p) * sin(r) + cos(r) * sin(p) * sin(y)) + face_y 
    x2 = size * (-cos(y) * sin(r)) + face_x
    y2 = size * (cos(p) * cos(r) - sin(p) * sin(y) * sin(r)) + face_y
    x3 = size * (sin(y)) + face_x
    y3 = size * (-cos(y) * sin(p)) + face_y

    # Draw base in red
    cv2.line(img, (int(face_x), int(face_y)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(face_x), int(face_y)), (int(x2),int(y2)),(0,0,255),3)
    cv2.line(img, (int(x2), int(y2)), (int(x2+x1-face_x),int(y2+y1-face_y)),(0,0,255),3)
    cv2.line(img, (int(x1), int(y1)), (int(x1+x2-face_x),int(y1+y2-face_y)),(0,0,255),3)
    # Draw pillars in blue
    cv2.line(img, (int(face_x), int(face_y)), (int(x3),int(y3)),(255,0,0),2)
    cv2.line(img, (int(x1), int(y1)), (int(x1+x3-face_x),int(y1+y3-face_y)),(255,0,0),2)
    cv2.line(img, (int(x2), int(y2)), (int(x2+x3-face_x),int(y2+y3-face_y)),(255,0,0),2)
    cv2.line(img, (int(x2+x1-face_x),int(y2+y1-face_y)), (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y)),(255,0,0),2)
    # Draw top in green
    cv2.line(img, (int(x3+x1-face_x),int(y3+y1-face_y)), (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y)),(0,255,0),2)
    cv2.line(img, (int(x2+x3-face_x),int(y2+y3-face_y)), (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y)),(0,255,0),2)
    cv2.line(img, (int(x3), int(y3)), (int(x3+x1-face_x),int(y3+y1-face_y)),(0,255,0),2)
    cv2.line(img, (int(x3), int(y3)), (int(x3+x2-face_x),int(y3+y2-face_y)),(0,255,0),2)

    return img


def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),4)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),4)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),4)

    return img


def get_pose_params_from_mat(mat_path):
    # This functions gets the pose parameters from the .mat
    # Annotations that come with the Pose_300W_LP dataset.
    mat = sio.loadmat(mat_path)
    # [pitch yaw roll tdx tdy tdz scale_factor]
    pre_pose_params = mat['Pose_Para'][0]
    # Get [pitch, yaw, roll, tdx, tdy]
    pose_params = pre_pose_params[:5]
    return pose_params

def get_ypr_from_mat(mat_path):
    # Get yaw, pitch, roll from .mat annotation.
    # They are in radians
    mat = sio.loadmat(mat_path)
    # [pitch yaw roll tdx tdy tdz scale_factor]
    pre_pose_params = mat['Pose_Para'][0]
    # Get [pitch, yaw, roll]
    pose_params = pre_pose_params[:3]
    return pose_params

def get_pt2d_from_mat(mat_path):
    # Get 2D landmarks
    mat = sio.loadmat(mat_path)
    pt2d = mat['pt2d']
    return pt2d

# batch*n
def normalize_vector(v):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))# batch
    gpu = v_mag.get_device()
    if gpu < 0:
        eps = torch.autograd.Variable(torch.FloatTensor([1e-8])).to(torch.device('cpu'))
    else:
        eps = torch.autograd.Variable(torch.FloatTensor([1e-8])).to(torch.device('cuda:%d' % gpu))
    v_mag = torch.max(v_mag, eps)
    v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
    v = v/v_mag
    return v
    
# u, v batch*n
def cross_product(u, v):
    batch = u.shape[0]
    #print (u.shape)
    #print (v.shape)
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
        
    out = torch.cat((i.view(batch,1), j.view(batch,1), k.view(batch,1)),1) #batch*3
        
    return out
        
    
#poses batch*6
#poses
def compute_rotation_matrix_from_ortho6d(poses):
    x_raw = poses[:,0:3] #batch*3
    y_raw = poses[:,3:6] #batch*3

    x = normalize_vector(x_raw) #batch*3
    z = cross_product(x,y_raw) #batch*3
    z = normalize_vector(z) #batch*3
    y = cross_product(z,x) #batch*3
        
    x = x.view(-1,3,1)
    y = y.view(-1,3,1)
    z = z.view(-1,3,1)
    matrix = torch.cat((x,y,z), 2) #batch*3*3
    return matrix


#input batch*4*4 or batch*3*3
#output torch batch*3 x, y, z in radiant
#the rotation is in the sequence of x,y,z
def compute_euler_angles_from_rotation_matrices(rotation_matrices):
    batch = rotation_matrices.shape[0]
    R = rotation_matrices
    sy = torch.sqrt(R[:,0,0]*R[:,0,0]+R[:,1,0]*R[:,1,0])
    singular = sy<1e-6
    singular = singular.float()
        
    x = torch.atan2(R[:,2,1], R[:,2,2])
    y = torch.atan2(-R[:,2,0], sy)
    z = torch.atan2(R[:,1,0],R[:,0,0])
    
    xs = torch.atan2(-R[:,1,2], R[:,1,1])
    ys = torch.atan2(-R[:,2,0], sy)
    zs = R[:,1,0]*0
        
    gpu = rotation_matrices.get_device()
    if gpu < 0:
        out_euler = torch.autograd.Variable(torch.zeros(batch,3)).to(torch.device('cpu'))
    else:
        out_euler = torch.autograd.Variable(torch.zeros(batch,3)).to(torch.device('cuda:%d' % gpu))
    out_euler[:,0] = x*(1-singular)+xs*singular
    out_euler[:,1] = y*(1-singular)+ys*singular
    out_euler[:,2] = z*(1-singular)+zs*singular
        
    return out_euler


def get_R(x,y,z):
    ''' Get rotation matrix from three rotation angles (radians). right-handed.
    Args:
        angles: [3,]. x, y, z angles
    Returns:
        R: [3, 3]. rotation matrix.
    '''
    # x
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(x), -np.sin(x)],
                   [0, np.sin(x), np.cos(x)]])
    # y
    Ry = np.array([[np.cos(y), 0, np.sin(y)],
                   [0, 1, 0],
                   [-np.sin(y), 0, np.cos(y)]])
    # z
    Rz = np.array([[np.cos(z), -np.sin(z), 0],
                   [np.sin(z), np.cos(z), 0],
                   [0, 0, 1]])

    R = Rz.dot(Ry.dot(Rx))
    return R


def resolve_yolo_weights(yolo_weights, yolo_version='yolov8', prefer_size='x', strict_version=False):
    """Resolve YOLO face-detector weights path with v8/v11 switch support.

        Priority:
      1) explicit yolo_weights if provided
      2) auto default by yolo_version
      3) fallback to existing counterpart if default file is missing

        When strict_version=True:
            - do not fallback to other versions/sizes
            - if requested weights cannot be found/downloaded, raise RuntimeError
    """
    version = str(yolo_version or 'yolov8').lower().strip()
    if version in ('8', 'v8', 'yolo8'):
        version = 'yolov8'
    elif version in ('11', 'v11', 'yolo11'):
        version = 'yolov11'

    size = str(prefer_size or 'x').lower().strip()
    if size not in ('n', 'x'):
        size = 'x'

    user_path = str(yolo_weights or '').strip()
    if user_path:
        return user_path

    base_dir = '/root/6DRepNet/facedat'
    default_map = {
        ('yolov8', 'n'): os.path.join(base_dir, 'yolov8n-face.pt'),
        ('yolov8', 'x'): os.path.join(base_dir, 'yolov8x-face.pt'),
        ('yolov11', 'n'): os.path.join(base_dir, 'yolo11n-face.pt'),
        ('yolov11', 'x'): os.path.join(base_dir, 'yolo11x-face.pt'),
    }

    def _download_to(url, dst):
        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            print(f"[YOLO] downloading weights: {url} -> {dst}")
            urllib.request.urlretrieve(url, dst)
            if os.path.exists(dst) and os.path.getsize(dst) > 0:
                print(f"[YOLO] downloaded: {dst}")
                return True
        except Exception as e:
            print(f"[YOLO] download failed from {url}: {e}")
        return False

    download_map = {
        ('yolov8', 'n'): [
            (os.path.join(base_dir, 'yolov8n-face.pt'), 'https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8n-face.pt'),
            (os.path.join(base_dir, 'yolov8n.pt'), 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt'),
        ],
        ('yolov8', 'x'): [
            (os.path.join(base_dir, 'yolov8x-face.pt'), 'https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8x-face.pt'),
            (os.path.join(base_dir, 'yolov8x.pt'), 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x.pt'),
        ],
        ('yolov11', 'n'): [
            (os.path.join(base_dir, 'yolo11n-face.pt'), 'https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolo11n-face.pt'),
            (os.path.join(base_dir, 'yolo11n.pt'), 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt'),
        ],
        ('yolov11', 'x'): [
            (os.path.join(base_dir, 'yolo11x-face.pt'), 'https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolo11x-face.pt'),
            (os.path.join(base_dir, 'yolo11x.pt'), 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt'),
        ],
    }

    primary = default_map.get((version, size), default_map[('yolov11', 'x')])
    if os.path.exists(primary):
        return primary

    # Auto download preferred weights if missing locally
    for dst_path, url in download_map.get((version, size), []):
        if os.path.exists(dst_path) and os.path.getsize(dst_path) > 0:
            return dst_path
        if _download_to(url, dst_path):
            return dst_path

    if bool(strict_version):
        raise RuntimeError(
            f"[YOLO] strict mode enabled: failed to resolve requested weights for "
            f"version={version}, size={size}. expected={primary}"
        )

    fallback_order = [
        default_map[(version, 'n' if size == 'x' else 'x')],
        default_map[('yolov8', size)],
        default_map[('yolov8', 'n' if size == 'x' else 'x')],
        default_map[('yolov11', size)],
        default_map[('yolov11', 'n' if size == 'x' else 'x')],
    ]
    for candidate in fallback_order:
        if os.path.exists(candidate):
            print(f"[YOLO] requested {version} but missing {primary}, fallback to {candidate}")
            return candidate

    return primary
