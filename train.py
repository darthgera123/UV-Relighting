import torch
import torch.nn as nn
import torch.optim as optim
from data import GaussianDataset, save_ply
from torch.utils.data import DataLoader
from torchvision import  transforms
from argparse import ArgumentParser
from imageio.v2 import imread,imwrite
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
from torchvision.utils import save_image
from model import UVRelit
from tqdm import tqdm
from unet import PortraitRelightingNet
from latent_model import LatentUnet,PosEnc
from unet_simple import UNet


def add_lights(diffuse,env_light):
    
    b,c,w,h = diffuse.shape
    env_light_flat = env_light.flatten(start_dim=1)
    env_light_rep = env_light_flat.repeat(1,w*h).view(b,-1,w,h)
    
    final = torch.cat((diffuse,env_light_rep),dim=1).float()
    return final

def prepare_input(diffuse,pos_enc=None):
    # print(diffuse)
    pos = diffuse['xyz']
    col = diffuse['features_dc']
    if pos_enc is not None:
        pos = pos_enc(pos) #[b,63,256,256]
    normalized_pos = (pos - pos.mean([2, 3], keepdim=True)) / (pos.std([2, 3], keepdim=True) + 1e-5)
    return torch.cat((pos,col),dim=1).float()
    # return col.float()

def gauss_loss(pred,gt,criterion):
    # params = ['xyz','scales','rotation','opacity','features_dc','features_rest']
    
    # coeff = [1,1,1,1,1,1]
    params = ['xyz','scales','rotation','opacity','features_dc','features_rest']
    
    # coeff = [0.01,1,1,1,0.1,0.1] #exp 1
    # coeff = [0.01,1,1,0.1,0.1,0.1] #exp 2
    coeff = [0.01,5,1,0.1,0.1,0.1] #exp 3
    loss = {}
    total_loss = 0
    for i,param in enumerate(params):
        loss[param] = criterion(pred[param],gt[param])
        total_loss += coeff[i]*loss[param]
    norm = torch.norm(pred['scales'], dim=1, keepdim=True).max()
    total_loss += 2e-3*norm
    return total_loss,loss

def to_cuda(pred):
    # params = ['xyz','scales','rotation','opacity','features_dc','features_rest']
    params = ['xyz','scales','rotation','opacity','features_dc','features_rest','rot_init']
    
    for i,param in enumerate(params):
        pred[param] = pred[param].cuda() 
    return pred

def to_cpu(pred):
    # params = ['xyz','scales','rotation','opacity','features_dc','features_rest']
    params = ['xyz','scales','rotation','opacity','features_dc','features_rest']
    for i,param in enumerate(params):
        pred[param] = pred[param].cpu().detach() 
    return pred

C0 = 0.28209479177387814
def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    return sh * C0 + 0.5

def norm_img_torch(image):
    min_value = 0.0
    max_value = 1.0

    image_min = torch.min(image)
    image_max = torch.max(image)
    
    # Normalize the image to the specified range
    img = (image - image_min) / (image_max - image_min) 
    return img

class RelitTrainer:
    def __init__(self, model, train_dataloader, val_dataloader,log_dir='runs/unet_experiment', save_dir='val_predictions', lr=0.001,epochs=500):
        
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        self.criterion = nn.L1Loss()  # Assuming a classification problem
        self.scale_criterion = nn.L1Loss(reduction='sum')
        self.pos_enc = None
        
        self.writer = SummaryWriter(log_dir=log_dir)
        self.save_dir = save_dir
        self.val_dir = os.path.join(self.save_dir,'renders')
        self.model_dir = os.path.join(self.save_dir,'checkpoint')
        
        os.makedirs(save_dir,exist_ok=True)
        os.makedirs(self.model_dir,exist_ok=True)
        os.makedirs(log_dir,exist_ok=True)
        os.makedirs(self.val_dir,exist_ok=True)

        self.model_name = model
        if self.pos_enc is None:
            in_ch = 6
        else: 
            in_ch = 66
        if self.model_name == 'uvrelit':
            self.model = UVRelit(in_ch=in_ch,out_ch=59)
            
        elif self.model_name == 'sipr':
            self.model = PortraitRelightingNet(light_size=(16,32))
        elif self.model_name == 'latent':
            self.model = LatentUnet(in_ch=6,out_ch=59)
        elif self.model_name == 'unet_simple':
            self.model = UNet(in_channels=6+1536,out_channels=59)
        else:
            raise NotImplementedError
        self.model = self.model.cuda()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=10, factor=0.5, min_lr=1e-6, verbose=True)
        self.lrelu = torch.nn.LeakyReLU(0.2)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.sigmoid
        self.tanh = torch.nn.Tanh()
        self.scaleparam = 1 #exp 1 was with 0.02
        self.rotation_init = torch.zeros((4,4,256,256), device="cuda") 

        

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader), desc=f"Epoch {epoch+1}", leave=False)
        
        for batch_idx, data_idx in progress_bar:
            self.optimizer.zero_grad()
            gt,light,diffuse,mask = data_idx
            gt,light,diffuse,mask = to_cuda(gt),light.cuda(),to_cuda(diffuse),mask.cuda()
            
            diffuse_input = prepare_input(diffuse,pos_enc=self.pos_enc) #(b,6,256,256)
            if self.model_name == 'uvrelit':
                relit_diffuse = add_lights(diffuse_input,light) #(b,6+1536,256,256)
                offset = self.model(relit_diffuse) #(b,59,256,256)
            elif self.model_name == 'sipr':
                offset = self.model(diffuse,light)
            elif self.model_name == 'latent':
                offset = self.model(diffuse_input,light)
            elif self.model_name == 'unet_simple':
                relit_diffuse = add_lights(diffuse_input,light)
                offset = self.model(relit_diffuse)
            
            # del_pos,del_scale,del_rot,del_op,del_col,del_sh = offset[:,0:3],offset[:,3:6],offset[:,6:10],offset[:,10:11],offset[:,11:14],offset[:,14:]
            del_pos = self.lrelu(offset[:,:3])*mask
            del_scale = self.sigmoid(offset[:,3:6])*mask
            del_scale = del_scale*self.scaleparam
            
            # del_rot = self.tanh(offset[:,6:10])
            del_rot = 2*self.sigmoid(offset[:,6:10]) -1
            # del_rot = del_rot*mask
            del_opacity = self.sigmoid(offset[:,10:11])*mask
            del_col = self.relu(offset[:,11:14])*mask
            del_sh = self.lrelu(offset[:,14:])
            
            
            relit_gauss = {}
            relit_gauss['xyz'] = diffuse['xyz'] + del_pos
            # relit_gauss['scales'] = diffuse['scales'] + del_scale
            relit_gauss['scales'] =  del_scale
            # relit_gauss['rotation'] = del_rot
            relit_gauss['rotation'] = diffuse['rot_init'] + del_rot
            relit_gauss['opacity'] =  del_opacity
            relit_gauss['features_dc'] = del_col
            relit_gauss['features_rest'] = del_sh
            # relit_gauss['features_rest'] = diffuse['features_rest'] + del_sh
            
            loss,param_loss = gauss_loss(relit_gauss,gt,self.criterion)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            # Update progress bar every iteration with the current loss
            progress_bar.set_postfix({'loss': '{:.6f}'.format(loss.item())})

            # Optionally, log to TensorBoard here as well
            self.writer.add_scalar('Loss/train', loss.item(), epoch * len(self.train_dataloader) + batch_idx)

        avg_loss = total_loss / len(self.train_dataloader)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Training Loss: {avg_loss:.6f}")

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_idx, data_idx in enumerate(self.val_dataloader):
                gt,light,diffuse,mask = data_idx
                gt,light,diffuse,mask = to_cuda(gt),light.cuda(),to_cuda(diffuse),mask.cuda()
                diffuse_input = prepare_input(diffuse,pos_enc=self.pos_enc) #(b,6,256,256)
                if self.model_name == 'uvrelit':
                    relit_diffuse = add_lights(diffuse_input,light)
                    offset = self.model(relit_diffuse)
                elif self.model_name == 'sipr':
                    offset = self.model(diffuse,light)
                elif self.model_name == 'latent':
                    offset = self.model(diffuse_input,light)
                elif self.model_name == 'unet_simple':
                    relit_diffuse = add_lights(diffuse_input,light)
                    offset = self.model(relit_diffuse)
                # del_pos,del_scale,del_rot,del_op,del_col,del_sh = offset[:,0:3],offset[:,3:6],offset[:,6:10],offset[:,10:11],offset[:,11:14],offset[:,14:]
                del_pos = self.lrelu(offset[:,:3])*mask
                del_scale = self.sigmoid(offset[:,3:6])*mask
                del_scale = del_scale*self.scaleparam
                
                # del_rot = self.tanh(offset[:,6:10])
                del_rot = 2*self.sigmoid(offset[:,6:10]) -1
                # del_rot = del_rot*mask
                del_opacity = self.sigmoid(offset[:,10:11])*mask
                del_col = self.relu(offset[:,11:14])*mask
                del_sh = self.lrelu(offset[:,14:])
            
                relit_gauss = {}
                relit_gauss['xyz'] = diffuse['xyz'] + del_pos
                

                relit_gauss['scales'] =  del_scale
                # relit_gauss['rotation'] =  del_rot
                relit_gauss['rotation'] = diffuse['rot_init'] + del_rot
                relit_gauss['opacity'] =  del_opacity
                relit_gauss['features_dc'] = del_col
                relit_gauss['features_rest'] = del_sh

                loss,param_loss = gauss_loss(relit_gauss,gt,self.criterion)
                total_loss += loss.item()
                if batch_idx == 0:  # Log validation images once per epoch
                    self.writer.add_images('Validation/Input Images', light, epoch)
                    # if outputs.shape[1] == 1:  # For single-channel outputs
                    #     outputs = outputs.repeat(1, 3, 1, 1)  # Make it 3-channel for visualization
                    pred_diff = relit_gauss['features_dc']
                    gt_diff = gt['features_dc']
                    
                    self.writer.add_images('Validation/Output Images', pred_diff, epoch)
                    self.writer.add_images('Validation/Target Images', gt_diff, epoch)
            # Log validation loss using TensorBoard
            
            self.writer.add_scalar('Loss/validation', total_loss / len(self.val_dataloader), epoch)
        self.scheduler.step(total_loss)
        print(f"Validation Loss: {total_loss / len(self.val_dataloader)}")

    
    def save_val_predictions(self, epoch):
        self.model.eval()
        with torch.no_grad():
            for batch_idx, data_idx in enumerate(self.val_dataloader):
                gt,light,diffuse,mask = data_idx
                gt,light,diffuse,mask = to_cuda(gt),light.cuda(),to_cuda(diffuse),mask.cuda()
                diffuse_input = prepare_input(diffuse,pos_enc=self.pos_enc) #(b,6,256,256)
                if self.model_name == 'uvrelit':
                    relit_diffuse = add_lights(diffuse_input,light)
                    offset = self.model(relit_diffuse)
                elif self.model_name == 'sipr':
                    
                    offset = self.model(diffuse,light)
                elif self.model_name == 'latent':
                    offset = self.model(diffuse_input,light)
                elif self.model_name == 'unet_simple':
                    relit_diffuse = add_lights(diffuse_input,light)
                    offset = self.model(relit_diffuse)
                # del_pos,del_scale,del_rot,del_op,del_col,del_sh = offset[:,0:3],offset[:,3:6],offset[:,6:10],offset[:,10:11],offset[:,11:14],offset[:,14:]
                del_pos = self.lrelu(offset[:,:3])*mask
                del_scale = self.sigmoid(offset[:,3:6])*mask
                del_scale = del_scale*self.scaleparam
                
                # del_rot = self.tanh(offset[:,6:10])
                del_rot = 2*self.sigmoid(offset[:,6:10]) -1
                # del_rot = del_rot*mask
                del_opacity = self.sigmoid(offset[:,10:11])*mask
                del_col = self.relu(offset[:,11:14])*mask
                del_sh = self.lrelu(offset[:,14:])
            
                relit_gauss = {}
                relit_gauss['xyz'] = diffuse['xyz'] + del_pos
                # relit_gauss['scales'] = diffuse['scales'] + del_scale
                relit_gauss['scales'] =  del_scale
                # relit_gauss['rotation'] =  del_rot
                relit_gauss['rotation'] = diffuse['rot_init'] + del_rot
                relit_gauss['opacity'] = del_opacity
                relit_gauss['features_dc'] = del_col
                relit_gauss['features_rest'] = del_sh
                # Ensure output is in CPU and detach it from the computation graph
                gt_col = gt['features_dc'].detach().cpu()
                pred_col = relit_gauss['features_dc'].detach().cpu()
                diffuse_col = diffuse['features_dc'].detach().cpu()
                gt_scale = norm_img_torch(gt['scales']).detach().cpu()
                pred_scale = norm_img_torch(relit_gauss['scales']).detach().cpu()
                diffuse_scale = norm_img_torch(diffuse['scales']).detach().cpu()
                for i in range(light.size(0)):
                    # Prepare the images
                    input_image = transforms.ToPILImage()(diffuse_col[i]).convert("RGB")
                    output_image = transforms.ToPILImage()(pred_col[i]).convert("RGB")
                    target_image = transforms.ToPILImage()(gt_col[i]).convert("RGB")

                    input_scale = transforms.ToPILImage()(diffuse_scale[i]).convert("RGB")
                    output_scale = transforms.ToPILImage()(pred_scale[i]).convert("RGB")
                    target_scale = transforms.ToPILImage()(gt_scale[i]).convert("RGB")
                    
                    # Convert PIL images to numpy arrays
                    input_np = np.array(input_image)
                    output_np = np.array(output_image)
                    target_np = np.array(target_image)
                    diff_np = np.abs(target_np/255.0-output_np/255.0)*255.0
                    diff_np = diff_np.astype('uint8')

                    input_scale_np = np.array(input_scale)
                    output_scale_np = np.array(output_scale)
                    target_scale_np = np.array(target_scale)
                    diff_scale_np = np.abs(target_scale_np/255.0-output_scale_np/255.0)*255.0
                    diff_scale_np = diff_scale_np.astype('uint8')
                    
                    # Stack images horizontally
                    combined_image = np.clip(np.hstack((output_np, target_np)),0,255)
                    combined_image_scale = np.clip(np.hstack((input_scale_np, output_scale_np, target_scale_np)),0,255)
                    
                    # Save the combined image
                    
                    file_name = os.path.join(self.val_dir, f'epoch_{epoch}_batch_{batch_idx}_diffuse_{i}.png')
                    file_name_scale = os.path.join(self.val_dir, f'epoch_{epoch}_batch_{batch_idx}_scale_{i}.png')
                    imwrite(file_name, combined_image)
                    imwrite(file_name_scale, combined_image_scale)
                return
    
    def save_checkpoint(self, epoch):
        path = os.path.join(self.model_dir,f'model_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load_checkpoint(self, path, optimizer=None):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        

    def save_predictions(self,dataloader,save_dir):
        os.makedirs(save_dir,exist_ok=True)
        pc_dir = os.path.join(save_dir,'point_cloud')
        os.makedirs(pc_dir,exist_ok=True)
        self.model.eval()
        with torch.no_grad():
            count = 0
            for batch_idx, data_idx in enumerate(tqdm(dataloader)):
                gt,light,diffuse,mask = data_idx
                gt,light,diffuse,mask = to_cuda(gt),light.cuda(),to_cuda(diffuse),mask.cuda()
                diffuse_input = prepare_input(diffuse,pos_enc=self.pos_enc) #(b,6,256,256)
                if self.model_name == 'uvrelit':
                    relit_diffuse = add_lights(diffuse_input,light)
                    offset = self.model(relit_diffuse)
                elif self.model_name == 'sipr':
                    offset = self.model(diffuse,light)
                elif self.model_name == 'latent':
                    offset = self.model(diffuse_input,light)
                elif self.model_name == 'unet_simple':
                    relit_diffuse = add_lights(diffuse_input,light)
                    offset = self.model(relit_diffuse)
                # del_pos,del_scale,del_rot,del_op,del_col,del_sh = offset[:,0:3],offset[:,3:6],offset[:,6:10],offset[:,10:11],offset[:,11:14],offset[:,14:]
                del_pos = self.lrelu(offset[:,:3])*mask
                del_scale = self.sigmoid(offset[:,3:6])*mask
                
                del_scale = del_scale*self.scaleparam
                
                # del_rot = self.tanh(offset[:,6:10])
                del_rot = 2*self.sigmoid(offset[:,6:10]) -1
                # del_rot = del_rot*mask
                del_opacity = self.sigmoid(offset[:,10:11])*mask
                del_col = self.relu(offset[:,11:14])*mask
                del_sh = self.lrelu(offset[:,14:])
                relit_gauss = {}
                relit_gauss['xyz'] = diffuse['xyz'] + del_pos
                # relit_gauss['xyz'] = gt['xyz']
                # relit_gauss['scales'] = diffuse['scales'] + del_scale
                relit_gauss['scales'] =  del_scale
                # relit_gauss['scales'] =  del_scale
                # relit_gauss['rotation'] = del_rot
                relit_gauss['rotation'] = diffuse['rot_init'] + del_rot
                relit_gauss['opacity'] =  del_opacity
                relit_gauss['features_dc'] = del_col
                relit_gauss['features_rest'] = del_sh
                # relit_gauss['features_rest'] = diffuse['features_rest'] + del_sh
                # relit_gauss['xyz'] = gt['xyz']
                # relit_gauss['scales'] = gt['scales']
                # relit_gauss['rotation'] = gt['rotation']
                # relit_gauss['opacity'] = gt['opacity']
                # relit_gauss['features_dc'] = gt['features_dc']
                # relit_gauss['features_rest'] = gt['features_rest']
                
                # Ensure output is in CPU and detach it from the computation graph
                # gt_col = SH2RGB(gt['features_dc']).detach().cpu()
                # pred_col = SH2RGB(relit_gauss['features_dc']).detach().cpu()
                # diffuse_col = SH2RGB(diffuse['features_dc']).detach().cpu()
                gt_col = gt['features_dc'].detach().cpu()
                pred_col = relit_gauss['features_dc'].detach().cpu()
                diffuse_col = diffuse['features_dc'].detach().cpu()
                gt_scale = norm_img_torch(gt['scales']).detach().cpu()
                pred_scale = norm_img_torch(relit_gauss['scales']).detach().cpu()
                diffuse_scale = norm_img_torch(diffuse['scales']).detach().cpu()
                for i in range(light.size(0)):
                    # Prepare the images
                    input_image = transforms.ToPILImage()(diffuse_col[i]).convert("RGB")
                    output_image = transforms.ToPILImage()(pred_col[i]).convert("RGB")
                    target_image = transforms.ToPILImage()(gt_col[i]).convert("RGB")


                    input_scale = transforms.ToPILImage()(diffuse_scale[i]).convert("RGB")
                    output_scale = transforms.ToPILImage()(pred_scale[i]).convert("RGB")
                    target_scale = transforms.ToPILImage()(gt_scale[i]).convert("RGB")
                    
                    # Convert PIL images to numpy arrays
                    input_np = np.array(input_image)
                    output_np = np.array(output_image)
                    target_np = np.array(target_image)
                    diff_np = np.abs(target_np/255.0-output_np/255.0)*255.0
                    diff_np = diff_np.astype('uint8')

                    input_scale_np = np.array(input_scale)
                    output_scale_np = np.array(output_scale)
                    target_scale_np = np.array(target_scale)
                    diff_scale_np = np.abs(target_scale_np/255.0-output_scale_np/255.0)*255.0
                    diff_scale_np = diff_scale_np.astype('uint8')
                    
                    # Stack images horizontally
                    combined_image = np.clip(np.hstack((output_np, target_np,diff_np)),0,255)
                    combined_image_scale = np.clip(np.hstack((output_scale_np, target_scale_np,diff_scale_np)),0,255)
                    # Save the combined image
                    env_name = gt['name'][i]
                    # file_name = os.path.join(save_dir, f'{str(batch_idx).zfill(3)}_{str(i).zfill(3)}.png')
                    file_name = os.path.join(save_dir, f'{env_name}.png')
                    # file_name_scale = os.path.join(save_dir, f'{str(batch_idx).zfill(3)}_{str(i).zfill(3)}_scale.png')
                    file_name_scale = os.path.join(save_dir, f'{env_name}_scale.png')
                    imwrite(file_name, combined_image)
                    imwrite(file_name_scale, combined_image_scale)

                    # path_name = os.path.join(pc_dir, f'{str(batch_idx).zfill(3)}_{str(i).zfill(3)}')+'_gt.ply'
                    path_name = os.path.join(pc_dir, f'{env_name}')+'_gt.ply'
                    save_ply(path_name,gt,index=i)

                    # path_name = os.path.join(pc_dir, f'{str(batch_idx).zfill(3)}_{str(i).zfill(3)}')+'_pred.ply'
                    path_name = os.path.join(pc_dir, f'{env_name}')+'_pred.ply'
                    save_ply(path_name,relit_gauss,index=i)


    def fit(self, epochs, validate_every_n_epochs=20,save_model =100):
        for epoch in range(epochs):
            self.train_epoch(epoch)
            if (epoch + 1) % validate_every_n_epochs == 0:
                self.validate(epoch)
                self.save_val_predictions(epoch)  # Save validation predictions on specified epochs
            if (epoch + 1) % save_model == 0:
                self.save_checkpoint(epoch)  # Save validation predictions on specified epochs
                
        self.writer.close()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--train_uv_maps", default="", help="specify calib file location")
    parser.add_argument("--train_env_maps", default="transforms.json", help="output path")
    parser.add_argument("--val_uv_maps", default="transforms.json", help="output path")
    parser.add_argument("--val_env_maps", default="transforms.json", help="output path")
    parser.add_argument("--envmap_list", default="transforms.json", help="output path")
    parser.add_argument("--diffuse", default="transforms.json", help="output path")
    parser.add_argument("--mask", default="mask.png", help="output path")
    parser.add_argument("--save_dir", default="transforms.json", help="output path")
    parser.add_argument("--checkpoint", default="transforms.json", help="output path")
    parser.add_argument("--lr",type=float, default=1e-3, help="output path")
    parser.add_argument("--epochs",type=int,default=100, help="create_images")
    parser.add_argument("--start",type=int,default=0, help="create_images")
    parser.add_argument("--end",type=int,default=100, help="create_images")
    parser.add_argument("--train",action='store_true', help="create_images")
    parser.add_argument("--skip_train",action='store_true', help="create_images")
    parser.add_argument("--network", default="uvrelit", help="output path")
    args = parser.parse_args()

    uv_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        ])

    env_transform = transforms.Compose([
            transforms.Resize((16, 32)),  # Downsample the envmap
            transforms.ToTensor(),
        ])
    if args.network == 'latent':
        resize = False
    else:
        resize = True
    train_data = GaussianDataset(uv_directory=args.train_uv_maps, env_directory=args.train_env_maps,
                            envmap_list = args.envmap_list,
                            diffuse=args.diffuse,resize=resize,start=args.start,end=args.end,mask=args.mask)
    train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=1)
    val_data = GaussianDataset(uv_directory=args.val_uv_maps, env_directory=args.val_env_maps,
                            envmap_list = args.envmap_list,
                            diffuse=args.diffuse,resize=resize,start=args.start,end=args.end,mask=args.mask)
    val_dataloader = DataLoader(val_data, batch_size=4, shuffle=False, num_workers=1)
    log_dir = os.path.join(args.save_dir,'logs')
    
    
    trainer = RelitTrainer(args.network, train_dataloader, val_dataloader,save_dir=args.save_dir,log_dir=log_dir,lr=args.lr)
    if args.train:
        trainer.fit(epochs=args.epochs,validate_every_n_epochs=20)
        val_save_path = os.path.join(args.save_dir,'val')
        trainer.save_predictions(val_dataloader,val_save_path)
        if not args.skip_train:
            train_save_path = os.path.join(args.save_dir,'train')
            trainer.save_predictions(train_dataloader,train_save_path)
    else:
        trainer.load_checkpoint(args.checkpoint)
        test_data = GaussianDataset(uv_directory=args.val_uv_maps, env_directory=args.val_env_maps,
                            envmap_list = args.envmap_list,
                            diffuse=args.diffuse,resize=resize,start=args.end,end=args.end+1,mask=args.mask)
        test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)
        test_save_path = os.path.join(args.save_dir,'val')
        train_save_path = os.path.join(args.save_dir,'train')
        # trainer.save_predictions(test_dataloader,test_save_path)
        if not args.skip_train:
            trainer.save_predictions(train_dataloader,train_save_path)

    
