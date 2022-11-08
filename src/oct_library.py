import torch
import matplotlib 
import numpy as np
import torchvision.transforms as T
# import albumentations as T
import matplotlib.pyplot as plt
import torch.nn.functional as F
import heyexReader as ep

from PIL import Image
from matplotlib import cm
from itertools import compress
from scipy.interpolate import interp1d
from skimage.restoration import denoise_tv_chambolle

class OCTProcessing:
    def __init__(self, oct_file, torchmodel):
        self.classes = ['BG', 'ILM', 'GCL', 'IPL', 'INL', 'OPL', 'ONL', 'ELM', 'EZ', 'BM']
        self.model = torchmodel
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        
        self.device = 'cuda'
        self.oct_file = oct_file
        self.oct_reader(self.oct_file)
        self.fovea_forward()

    def __len__(self):
        return len(self.oct)
        
    def oct_reader(self, oct_file):
        self.oct = ep.volFile(oct_file)
        self.bscan_fovea = self.oct.oct[self.oct.fileHeader['numBscan']//2, :, :]


    def predict(self, model, x_image):
        transforms = T.Normalize(mean=0.1338, std=0.1466) # CONTROL: 0.0389,  0.1036,  # FULL: 0.1338, 0.1466
        image = np.expand_dims(x_image, axis=-1)
        image = torch.tensor(image, dtype=torch.float, device=self.device)
        image = transforms(image)
        image = torch.permute(image, (2, 0, 1))
        
        print(image.max(), image.min())
        image = image.unsqueeze(0)
        y_pred = model(image)
        y_pred = F.softmax(y_pred, dim=1)
        y_pred = torch.argmax(y_pred, dim=1)
        y_pred = y_pred.squeeze(0).detach().cpu().numpy()

        return y_pred

    def get_segmentation(self, img, gamma, alpha, imgw, imgh):
        img = gray_gamma(img, gamma=gamma)
        img = tv_denoising(img, alpha=alpha)

        img_h, img_w = img.shape
        
        image_x = F.interpolate(torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float(), (imgh, imgw), mode='bilinear', align_corners=False).squeeze().numpy()
        pred = self.predict(self.model, image_x)
        pred = F.interpolate(torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).float(), (img_h, img_w), mode='nearest').squeeze().numpy()

        shape_1 = (pred.shape[0], pred.shape[1], 3)
        pred_rgb = np.zeros(shape=shape_1, dtype='uint8')

        norm = matplotlib.colors.Normalize(vmin=0, vmax=pred.max())
        for idx in range(1, int(pred.max())+1):
            pred_rgb[..., 0] = np.where(pred == idx, cm.hsv(norm(idx), bytes=True)[0], pred_rgb[..., 0])
            pred_rgb[..., 1] = np.where(pred == idx, cm.hsv(norm(idx), bytes=True)[1], pred_rgb[..., 1])
            pred_rgb[..., 2] = np.where(pred == idx, cm.hsv(norm(idx), bytes=True)[2], pred_rgb[..., 2])

        pred_overlay = Image.fromarray(pred_rgb)
        img_overlay = Image.fromarray(img)
        pred_overlay = pred_overlay.convert("RGBA")
        img_overlay = img_overlay.convert("RGBA")
        overlay = Image.blend(img_overlay, pred_overlay, 0.3)
        overlay = np.array(overlay)
        return pred, pred_rgb, overlay

    def get_layer_binary_mask(self, pred, layer):
        binary = np.where(pred == self.classes.index(layer), 1, 0)
        size = 1
        for i in range(size, binary.shape[1], size):
            col = binary[:, i - size:i]
            if 1 in col:
                place = np.max(np.where(col)[0])
                binary[place, i - size:i] = 0
        return binary

    def get_individual_layers_segmentation(self, layer: list):
        classmap_bool = list(compress(self.classes, layer))
        segmented = np.zeros(self.bscan_fovea.shape)
        for l in classmap_bool:
            binary = np.where(self.pred_class_map == self.classes.index(l), 1, 0)
            multi = np.multiply(binary, self.bscan_fovea)
            segmented = np.add(self.segmented, multi)
        return segmented

    def plot_selected_layers(self):
        fig, ax = plt.subplots(nrows=1, ncols=1, dpi=200, figsize=(14,10), gridspec_kw={'width_ratios': [1]}, frameon=False)
        ax.set_xlabel('B-Scan (X)', fontsize=14, weight="bold")
        ax.set_ylabel('A-Scan (Y)', fontsize=14, weight="bold")
        ax.tick_params(labelsize=12)
        ax.tick_params(labelsize=12)
        ax.imshow(self.segmented, cmap='gray')

    def plot_overlay_oct_segmentation(self):
        fig, ax = plt.subplots(nrows=1, ncols=1, dpi=200, figsize=(14,10), gridspec_kw={'width_ratios': [1]}, frameon=False)
        ax.set_xlabel('B-Scan (X)', fontsize=14, weight="bold")
        ax.set_ylabel('A-Scan (Y)', fontsize=14, weight="bold")
        ax.tick_params(labelsize=12)
        ax.tick_params(labelsize=12)
        ax.imshow(self.overlay)

    def plot_segmentation_full(self):
        fig, ax = plt.subplots(nrows=1, ncols=1, dpi=200, figsize=(14,10), gridspec_kw={'width_ratios': [1]}, frameon=False)
        ax.set_xlabel('B-Scan (X)', fontsize=14, weight="bold")
        ax.set_ylabel('A-Scan (Y)', fontsize=14, weight="bold")
        ax.tick_params(labelsize=12)
        ax.tick_params(labelsize=12)
        ax.imshow(self.segmented_total, cmap='gray')

    def plot_slo_fovea(self):
        fig, ax = plt.subplots(nrows=1, ncols=2, dpi=200, figsize=(25,10), gridspec_kw={'width_ratios': [1, 2]}, frameon=False)
        ax[0].imshow(self.oct.irslo, cmap='gray')
        # self.oct.plot(localizer=True, bscan_positions=True, ax=ax[0])
        ax[1].imshow(self.bscan_fovea, cmap='gray')
        ax[1].set_xlabel('B-Scan (X)', fontsize=14, weight="bold")
        ax[1].set_ylabel('A-Scan (Y)', fontsize=14, weight="bold")
        ax[0].set_xlabel('B-Scan (X)', fontsize=14, weight="bold")
        ax[0].set_ylabel('Volume (Z)', fontsize=14, weight="bold")
        ax[0].tick_params(labelsize=12)
        ax[0].tick_params(labelsize=12)
        ax[1].tick_params(labelsize=12)
        ax[1].tick_params(labelsize=12)   

    def fovea_forward(self, gamma=1.5, alpha=0.02, imgh=512, imgw=512):

        self.pred_class_map, self.pred_rgb, self.overlay = self.get_segmentation(self.bscan_fovea, gamma=gamma, alpha=alpha, imgh=imgh, imgw=imgw)

        self.binary_total = np.where(np.logical_and(self.pred_class_map.astype('float') <= 10, self.pred_class_map.astype('float') > 0), 1, 0)

        self.segmented_total = np.multiply(self.binary_total, self.bscan_fovea)


def gray_gamma(img, gamma):
    gray = img / 255.
    out = np.array(gray ** gamma)
    out = 255*out
    return out.astype('uint8')

def tv_denoising(img, alpha):
    gray = img / 255.
    out = denoise_tv_chambolle(gray, weight=alpha)
    out = out * 255
    return out.astype('uint8')

def denoising_1D_TV(Y, lamda):
    N = len(Y)
    X = np.zeros(N)

    k, k0, kz, kf = 0, 0, 0, 0
    vmin = Y[0] - lamda
    vmax = Y[0] + lamda
    umin = lamda
    umax = -lamda

    while k < N:
        
        if k == N - 1:
            X[k] = vmin + umin
            break
        
        if Y[k + 1] < vmin - lamda - umin:
            for i in range(k0, kf + 1):
                X[i] = vmin
            k, k0, kz, kf = kf + 1, kf + 1, kf + 1, kf + 1
            vmin = Y[k]
            vmax = Y[k] + 2 * lamda
            umin = lamda
            umax = -lamda
            
        elif Y[k + 1] > vmax + lamda - umax:
            for i in range(k0, kz + 1):
                X[i] = vmax
            k, k0, kz, kf = kz + 1, kz + 1, kz + 1, kz + 1
            vmin = Y[k] - 2 * lamda
            vmax = Y[k]
            umin = lamda
            umax = -lamda
            
        else:
            k += 1
            umin = umin + Y[k] - vmin
            umax = umax + Y[k] - vmax
            if umin >= lamda:
                vmin = vmin + (umin - lamda) * 1.0 / (k - k0 + 1)
                umin = lamda
                kf = k
            if umax <= -lamda:
                vmax = vmax + (umax + lamda) * 1.0 / (k - k0 + 1)
                umax = -lamda
                kz = k
                
        if k == N - 1:
            if umin < 0:
                for i in range(k0, kf + 1):
                    X[i] = vmin
                k, k0, kf = kf + 1, kf + 1, kf + 1
                vmin = Y[k]
                umin = lamda
                umax = Y[k] + lamda - vmax
                
            elif umax > 0:
                for i in range(k0, kz + 1):
                    X[i] = vmax
                k, k0, kz = kz + 1, kz + 1, kz + 1
                vmax = Y[k]
                umax = -lamda
                umin = Y[k] - lamda - vmin
                
            else:
                for i in range(k0, N):
                    X[i] = vmin + umin * 1.0 / (k - k0 + 1)
                break

    return X
