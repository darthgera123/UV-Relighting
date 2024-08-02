import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from imageio.v2 import imread,imwrite
import cv2
from PIL import Image
from argparse import ArgumentParser
from plyfile import PlyData, PlyElement
import re
"""
We will load gaussian parameters here
"""
def sigmoid(x):
    return 1/(1+np.exp(-x))

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def normalize(x,axis=1,eps=1e-12):
    norm = np.linalg.norm(x,axis=axis,keepdims=True)
    return x/(norm+eps)
def extract_light(path):
    """
    This function extracts the '<light>' part from a given path string.
    
    Args:
    path (str): The full path string from which to extract the '<light>' part.
    
    Returns:
    str: The extracted '<light>' part of the path.
    """
    # Regular expression to match the pattern around '<light>'
    match = re.search(r'/pose_01/(.*?)/point_cloud/', path)
    if match:
        return match.group(1)  # Return the captured group which is '<light>'
    else:
        return "No light part found" 
def load_ply(path,sh_degree=3):
    max_sh_degree = sh_degree
    
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    f_dc = np.zeros((xyz.shape[0], 3, 1))
    f_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    f_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    f_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
    features_dc = f_dc

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    assert len(extra_f_names)==3*(max_sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
    
    # scale_init_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("init_scale_")]
    # scale_init_names = sorted(scale_init_names, key = lambda x: int(x.split('_')[-1]))
    
    # scales_init = np.zeros((xyz.shape[0], len(scale_init_names)))
    
    # for idx, attr_name in enumerate(scale_init_names):
        
    #     scales_init[:, idx] = np.asarray(plydata.elements[0][attr_name])
    
    gauss_feat = {}

    rot_init = np.zeros((xyz.shape[0], 4))
    rot_init[:, 0] = 1

    # gauss_feat['xyz'] = torch.tensor(xyz, dtype=torch.float, device="cuda")
    # gauss_feat['features_dc'] = torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()
    # gauss_feat['features_rest'] = torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()
    # gauss_feat['opacity'] = torch.tensor(opacities, dtype=torch.float, device="cuda")
    # gauss_feat['scaling'] = torch.tensor(scales, dtype=torch.float, device="cuda")
    # gauss_feat['rotation'] = torch.tensor(rots, dtype=torch.float, device="cuda")
    gauss_feat['xyz'] = xyz.reshape(256,256,3).transpose(2,0,1)
    gauss_feat['opacity'] = sigmoid(opacities).reshape(256,256,1).transpose(2,0,1)
    # gauss_feat['rotation'] = normalize(rots).reshape(256,256,4).transpose(2,0,1)
    gauss_feat['rotation'] = rots.reshape(256,256,4).transpose(2,0,1)
    gauss_feat['rot_init'] = rot_init.reshape(256,256,4).transpose(2,0,1)
    gauss_feat['scales'] = np.exp(scales).reshape(256,256,3).transpose(2,0,1)
    # if scales_init.shape == (256*256,3):
    #     gauss_feat['scales_init'] = scales_init.reshape(256,256,3).transpose(2,0,1)
    # else:
    #     gauss_feat['scales_init'] = scales_init
    # gauss_feat['scales'] = scales.reshape(256,256,3).transpose(2,0,1)
    
    gauss_feat['features_dc'] = SH2RGB(features_dc).reshape(256,256,3).transpose(2,0,1)
    # gauss_feat['features_dc'] = SH2RGB(features_dc).reshape(256,256,3).transpose(2,0,1)
    gauss_feat['features_rest'] = features_extra.reshape(256,256,45).transpose(2,0,1)
    gauss_feat['name'] = extract_light(path)
    return gauss_feat

def construct_list_of_attributes():
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        # for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
        for i in range(1*3):
            l.append('f_dc_{}'.format(i))
        for i in range(3*15):
        # for i in range(1*45):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(3):
            l.append('scale_{}'.format(i))
        for i in range(4):
            l.append('rot_{}'.format(i))
        return l




def save_ply(path,relit,sh_degree=3,index=0):
    os.makedirs(os.path.dirname(path),exist_ok=True)
    xyz = relit['xyz'][index].detach().cpu().numpy().transpose(1,2,0).reshape(-1,3)
    
    normals = np.zeros_like(xyz)
    # scale = torch.log(relit['scales'][index]+1e-8).detach().cpu().numpy().transpose(1,2,0).reshape(-1,3)
    scale = inverse_sigmoid(relit['scales'][index]+1e-8).detach().cpu().numpy().transpose(1,2,0).reshape(-1,3)
    # scale = relit['scales'][index].detach().cpu().numpy().transpose(1,2,0).reshape(-1,3)
    rotation = relit['rotation'][index].detach().cpu().numpy().transpose(1,2,0).reshape(-1,4)
    # f_dc = Nx3x1
    f_dc = RGB2SH(relit['features_dc'][index]).detach().cpu().numpy().transpose(1,2,0).reshape(-1,3)
    
    # f_rest = Nx3x15
    f_rest = relit['features_rest'][index].detach().cpu().numpy().transpose(1,2,0).reshape(-1,3*((sh_degree + 1) ** 2 - 1))
    opacities = inverse_sigmoid(relit['opacity'][index]).reshape(-1,1).detach().cpu().numpy()
    # print(xyz.shape,scale.shape)
    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes()]
    attributes = np.concatenate((xyz, normals,f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)

    np_scale = relit['scales'][index].detach().cpu().numpy().transpose(1,2,0).reshape(-1,3)
    scale_path = path.split('.')[0] + '_scale.npy'
    np.save(scale_path,np_scale)

def resize_max_pool(image, pool_size=(16, 16)):
    # Assuming image shape is (height, width, channels)
    img_height, img_width, channels = image.shape
    pool_height, pool_width = pool_size
    
    # Define output dimensions
    out_height = img_height // pool_height
    out_width = img_width // pool_width
    
    image_reshaped = image.reshape(out_height, pool_height, out_width, pool_width, channels)
    
    pooled_image = np.max(image_reshaped, axis=(1, 3))
    return pooled_image

def parse_args():
    parser =  ArgumentParser(description="convert calib file to nerf format transforms.json")
    parser.add_argument("--uv_maps", default="", help="specify calib file location")
    parser.add_argument("--env_maps", default="transforms.json", help="output path")
    parser.add_argument("--mask", default="transforms.json", help="output path")
    parser.add_argument("--diffuse", default="transforms.json", help="output path")
    parser.add_argument("--envmap_list", default="transforms.json", help="output path")
    parser.add_argument("--create_alpha",action='store_true', help="create_images")
    parser.add_argument("--scale", default=1,type=int, help="scale")
    args = parser.parse_args()
    return args

def norm_img(image):
    min_value = 0.0
    max_value = 1.0

    image_min = np.min(image)
    image_max = np.max(image)
    
    # Normalize the image to the specified range
    img = (image - image_min) / (image_max - image_min) 
    return img

def env(path,resize=True):
    image = imread(path)
    if resize:
        # image = cv2.resize(image, (32,16), interpolation=cv2.INTER_AREA)
        image = resize_max_pool(image,pool_size=(16,16))
    image = norm_img(image)
    if image.dtype == 'uint8':
        resized_image = torch.from_numpy(np.array(image)) / 255.0
    else:
        resized_image = torch.from_numpy(np.array(image)) 
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def read_env(path,start=0,end=100):
    with open(path, 'r') as file:
    # Read all lines from the file and store them in a list
        lines = [line.strip() for line in file]
    return lines[start:end]

"""
Load base geometry features
Load tupple (gauss,light) features
"""
C0 = 0.28209479177387814
def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    return sh * C0 + 0.5
class GaussianDataset(Dataset):
    def __init__(self, uv_directory, env_directory,envmap_list, diffuse=None,resize=True,mask=None,start=0,end=10):
        """
        Args:
            uv_directory (string): Directory with all the UV map images.
            env_directory (string): Directory with all the envmap images.
            uv_transform (callable, optional): Optional transform to be applied on a UV map.
            env_transform (callable, optional): Optional transform to be applied on an envmap.
        """
        self.uv_directory = uv_directory
        self.env_directory = env_directory
        self.resize = resize
        self.images = [f.split('.')[0] for f in os.listdir(uv_directory) if os.path.isfile(os.path.join(uv_directory, f))]
        self.diffuse_geometry = load_ply(diffuse)
        self.envmap_list = envmap_list
        self.envmaps = read_env(self.envmap_list,start=start,end=end)
        if mask is not None:
            self.mask = imread(os.path.join(mask))/255.0
            self.mask = self.mask.reshape(1,256,256)
        else:
            self.mask = None
        
        
    def __len__(self):
        return len(self.envmaps)

    def __getitem__(self, idx):
        uv_params_name = os.path.join(self.uv_directory,self.envmaps[idx],'point_cloud/iteration_20000/point_cloud.ply')
        env_img_name = os.path.join(self.env_directory, self.envmaps[idx])+'.exr'
        uv_params = load_ply(uv_params_name)
        
        
        
        env_image = env(env_img_name,resize=self.resize)
        
        return uv_params, env_image, self.diffuse_geometry, self.mask
    
if __name__ == "__main__":
    args = parse_args()
    # uv_transform = transforms.Compose([
    # transforms.Resize((256, 256)),
    # transforms.ToTensor(),
    # ])

    # env_transform = transforms.Compose([
    #     transforms.Resize((16, 32)),  # Downsample the envmap
    #     transforms.ToTensor(),
    # ])

    # # Assuming your UV maps and envmaps are stored in 'path/to/uvmaps' and 'path/to/envmaps' respectively
    dataset = GaussianDataset(uv_directory=args.uv_maps, env_directory=args.env_maps,
                            envmap_list = args.envmap_list,
                            diffuse=args.diffuse,resize=True,mask=args.mask)
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1)

    for i, (uv_maps, env_maps, diffuse,mask) in enumerate(dataloader):
    # uv_maps and env_maps are your batches of UV maps and envmaps
        print(uv_maps['features_dc'].shape, env_maps.shape, diffuse['xyz'].shape,mask.shape)  # Expect [batch_size, 3, 256, 256] for UV maps and [batch_size, 3, 16, 32] for envmaps
        diff_image = diffuse['features_dc'][0].detach().cpu().numpy().transpose(1,2,0)
        rot_image_diffuse = diffuse['scales'][0].detach().cpu().numpy().transpose(1,2,0)
        print(uv_maps['name'][0])
        # rot_image_diffuse = np.exp(rot_image_diffuse)
        pos_img = diffuse['scales'][1].detach().cpu().numpy().transpose(1,2,0)
        print(pos_img.min(),pos_img.max())
        mask_img  = mask[0].detach().cpu().numpy().transpose(1,2,0)
        # pos_img = np.log(pos_img)*mask_img
        pos_img = (pos_img - pos_img.min()) / (pos_img.max() - pos_img.min())
        # diff_image = np.clip(diff_image,0,1).reshape(256,256,3)
        # print(diff_image.min(),diff_image.max())
        # diff_image = (diff_image*255).astype('uint8')
        # relit_image = uv_maps['features_dc'][0].detach().cpu().numpy().transpose(1,2,0)
        
        # relit_image = np.clip(relit_image,0,1).reshape(256,256,3)
        # relit_image = (relit_image*255).astype('uint8')
        # rot_image_relit = uv_maps['scales'][0].detach().cpu().numpy().transpose(1,2,0)
        # rot_image_relit = np.exp(rot_image_relit)
        # print(rot_image_relit.min(),rot_image_relit.max())
        # np_img = np.hstack([relit_image,diff_image])
        imwrite(f'diff_pos.png',(pos_img*255).astype('uint8'))
        cv2.imwrite(f'mask_img.png',(mask_img*255).astype('uint8'))
        env_map = env_maps[0].detach().cpu().numpy().transpose(1,2,0)
        env_map = (env_map*255).astype('uint8')
        imwrite(f'env.png',env_map.astype('uint8'))
        
        save_ply(path='./point_cloud.ply',relit=diffuse,index = 0)
        if i == 1:  # just to break after one batch for demonstration
            break
    # gauss_diffuse = load_ply(args.diffuse)
    # print(gauss_diffuse['xyz'])