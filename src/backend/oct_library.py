import torch
import matplotlib 
import numpy as np
import eyepy as ep
import albumentations as T
import matplotlib.pyplot as plt
import torch.nn.functional as F

from PIL import Image
from matplotlib import cm
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from patchify import patchify, unpatchify
from skimage.restoration import denoise_tv_chambolle

class OCTProcessing:
    def __init__(self, oct_file, torchmodel):
        self.classes = ['BG', 'ILM', 'GCL', 'IPL', 'INL', 'OPL', 'ONL', 'ELM', 'EZ', 'BM']
        self.model = torchmodel
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        
        self.oct_file = oct_file
        self.oct_reader(self.oct_file)
        self.bscan_metadata()

    def __len__(self):
        return len(self.oct)
        
    def oct_reader(self, oct_file):
        self.oct = ep.import_heyex_vol(oct_file)
        self.scale_y = self.oct.meta.as_dict()['scale_y']
        self.scale_x = self.oct.meta.as_dict()['scale_x']
        self.visit_date = self.oct.meta.as_dict()['visit_date']
        self.laterality = self.oct.meta.as_dict()['laterality']
        self.loc_fovea = self.oct.meta.as_dict()['bscan_meta'][self.__len__() // 2]['start_pos'][1] // self.scale_x
        self.fovea_xstart = self.oct.meta.as_dict()['bscan_meta'][self.__len__() // 2]['start_pos'][0] // self.scale_x
        self.fovea_xstop = self.oct.meta.as_dict()['bscan_meta'][self.__len__() // 2]['end_pos'][0] // self.scale_x
        self.bscan_fovea = self.oct[len(self.oct)//2].data

    def bscan_metadata(self):
        print('No. of Scans: ', len(self.oct))
        print('Y-Fovea: ', self.loc_fovea)
        print('ScaleY: ', self.scale_y)
        print('ScaleX: ', self.scale_x)
        print('Visit Date: ', self.visit_date)
        print('Laterality: ', self.laterality)
        self.bscan_start_pos = self.oct.meta.as_dict()['bscan_meta'][len(self.oct)//2]['start_pos'][0]
        self.bscan_end_pos = self.oct.meta.as_dict()['bscan_meta'][len(self.oct)//2]['end_pos'][0]

    def predict(self, model, x_image):
        transforms = T.Compose([
                    T.Normalize(mean=0.1338, std=0.1466) # CONTROL: 0.0389,  0.1036,  # FULL: 0.1338, 0.1466
        ])

        image = np.expand_dims(x_image, axis=-1)

        image = transforms(image=image)
        image = image['image'].transpose((2, 0, 1))
        image = torch.tensor(image, dtype=torch.float, device=self.device)
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
        overlay = Image.blend(img_overlay, pred_overlay, 0.5)
        overlay = np.array(overlay)
        return pred, pred_rgb, overlay

    def get_reference_EZ_segmentation(self, msk, offset=2):
        mask_ez = np.where(msk == self.classes.index('EZ'), 1, 0)
        pos_ez = np.where(mask_ez)
        try:
            xmin = np.min(pos_ez[1][np.nonzero(pos_ez[1])])
        except:
            xmin = 0
        try:
            xmax = np.max(pos_ez[1][np.nonzero(pos_ez[1])])
        except:
            xmax = 10       
        mask_ilm = np.where(msk == self.classes.index('ILM'), 1, 0)
        pos_ilm = np.where(mask_ilm)
        ymin = np.min(pos_ilm[0][np.nonzero(pos_ilm[0])])

        mask_bm = np.where(msk == self.classes.index('BM'), 1, 0)
        pos_bm = np.where(mask_bm)
        ymax = np.max(pos_bm[0][np.nonzero(pos_bm[0])])

        if ymin < offset:
            ymin = 0
        else:
            ymin = ymin-offset

        if (ymax + offset) > msk.shape[0]:
            ymax = msk.shape[0]
        else:
            ymax = ymax+offset
        return ymin, ymax, xmin, xmax\

    def get_limit(self, binary_mask, side):
        size = 1
        lim = []
        for i in range(size, binary_mask.shape[1], size):
            col = binary_mask[:, i - size:i]
            if 1 in col:
                if side == 'max':
                    lim.append(np.max(np.where(col)[0]))
                if side == 'min':
                    lim.append(np.min(np.where(col)[0]))
            else:
                lim.append(float('nan'))
        lim = np.array(lim)
        return lim

    def get_layer_binary_mask(self,sample_pred, layer='EZ', offset=0):
        binary = np.where(sample_pred == self.classes.index(layer), 1, 0)
        size = 1
        if offset > 0:
            for off in range(offset):
                for i in range(size, binary.shape[1], size):
                    col = binary[:, i - size:i]
                    if 1 in col:
                        place = np.max(np.where(col)[0])
                        binary[place, i - size:i] = 0
        return binary

    def get_individual_layers_segmentation(self, layer):
        self.binary = self.get_layer_binary_mask(self.sample_pred, layer=layer, offset=0)
        self.segmented = np.multiply(self.binary, self.sample_bscan)

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
        self.oct.plot(localizer=True, bscan_positions=True, ax=ax[0])
        ax[1].imshow(self.oct[len(self.oct)//2].data, cmap='gray')
        ax[0].scatter(20+self.fovea_xstart, self.loc_fovea, c='red', s=50)
        ax[0].scatter(self.oct[len(self.oct)//2].data.shape[1]-20+self.fovea_xstart, self.loc_fovea, c='red', s=50)
        ax[1].set_xlabel('B-Scan (X)', fontsize=14, weight="bold")
        ax[1].set_ylabel('A-Scan (Y)', fontsize=14, weight="bold")
        ax[0].set_xlabel('B-Scan (X)', fontsize=14, weight="bold")
        ax[0].set_ylabel('Volume (Z)', fontsize=14, weight="bold")
        ax[0].tick_params(labelsize=12)
        ax[0].tick_params(labelsize=12)
        ax[1].tick_params(labelsize=12)
        ax[1].tick_params(labelsize=12)   

    def fovea_forward(self, gamma, alpha, imgh, imgw):
        self.pixel_2_mm2 = self.scale_x * self.scale_y

        self.pred_class_map, self.pred_rgb, self.overlay = self.get_segmentation(self.bscan_fovea, gamma=gamma, alpha=alpha, imgh=imgh, imgw=imgw)

        ymin, ymax, xmin, xmax = self.get_reference_EZ_segmentation(msk=self.pred_class_map, offset=2)
        self.references = [xmin, xmax, ymin, ymax]

        self.sample_bscan = self.bscan_fovea[ymin:ymax,xmin:xmax]
        # p = np.percentile(self.sample_bscan, 95)
        # self.sample_bscan = self.sample_bscan / p
        self.sample_pred = self.pred_class_map[ymin:ymax,xmin:xmax]
        self.sample_pred_rgb = self.pred_rgb[ymin:ymax,xmin:xmax]
        self.sample_overlay = self.overlay[ymin:ymax,xmin:xmax]

        self.binary_total = np.where(np.logical_and(self.pred_class_map.astype('float') <= 10, self.pred_class_map.astype('float') > 0), 1, 0)

        self.segmented_total = np.multiply(self.binary_total, self.bscan_fovea)

    def volume_forward(self, tv_smooth=False, plot=False, imgh=496, imgw=496, bscan_positions=True):
        X_MINS = []
        X_MAX = []
        Y_POS = []
        delta_ez_lim = []
        for idx in range(len(self.oct)):
            bscan = self.oct[idx].data
            try:
                xstart = self.oct.meta.as_dict()['bscan_meta'][idx]['start_pos'][0]//self.oct.meta.as_dict()['scale_x']
                xstop = self.oct.meta.as_dict()['bscan_meta'][idx]['end_pos'][0]//self.oct.meta.as_dict()['scale_x']
                pred_class_map, _, _ = self.get_segmentation(bscan, gamma=2, alpha=0.03, imgh=imgh, imgw=imgw)
                _, _, xmin, xmax = self.get_reference_EZ_segmentation(pred_class_map, 32)
                if (xmax - xmin) < 50:
                    xmin = 0
                    xmax = 10
                if xmin != 0 and xmax != 10:
                    delta_ez_lim.append(np.abs(xmax * self.scale_x - xmin * self.scale_x))
                    # X_MINS.append(xmin)
                    # X_MAX.append(xmax)
                    X_MINS.append(xmin + xstart)
                    X_MAX.append(xmax + (xstart))
                    Y_POS.append(self.oct.meta.as_dict()['bscan_meta'][idx]['start_pos'][1] // self.oct.meta.as_dict()['scale_x'])
            except Exception as exc:
                print(exc)
                continue
        if plot:
            plt.figure(dpi=200, figsize=(8,8), frameon=False)
            self.oct.plot(localizer=True, bscan_positions=bscan_positions)
            plt.plot(X_MINS, Y_POS, '--', c='orangered', linewidth=1, label='X-min EZ Limit')
            plt.plot(X_MAX, Y_POS, '--', c='orange', linewidth=1, label='X-max EZ Limits')
            plt.scatter(X_MINS, Y_POS, c='red', s=10)
            plt.scatter(X_MAX, Y_POS, c='red', s=10)
            if tv_smooth:
                ynew1 = np.linspace(0, np.array(Y_POS).max(), num=300)
                f1 = interp1d(Y_POS, X_MAX, kind='linear', fill_value="extrapolate")
                YNEW1 = denoising_1D_TV(f1(ynew1), 10)
                plt.plot(YNEW1, ynew1, '-', c='cornflowerblue', linewidth=1, label='TV X-max EZ Limit')
                ynew2 = np.linspace(0, np.array(Y_POS).max(), num=300)
                f2 = interp1d(Y_POS, X_MINS, kind='linear', fill_value="extrapolate")
                YNEW2 = denoising_1D_TV(f2(ynew2), 10)
                plt.plot(YNEW2, ynew2, '-', c='blue', linewidth=1, label='TV X-min EZ Limit')
                plt.legend(loc='best')
            plt.tick_params(labelsize=12)
            plt.xticks([])
            plt.yticks([])
            # plt.ylim([np.array(Y_POS).max(), 0])
            # plt.xlim([0, np.array(X_MAX).max()])
            plt.tight_layout()
        return Y_POS, delta_ez_lim

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
