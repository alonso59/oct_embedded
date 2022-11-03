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
    def __init__(self, oct_file, torchmodel, dataset='bonn', mode='large'):
        self.classes = ['BG', 'EZ', 'OPL', 'ELM', 'BM', 'ONL', 'GCL', 'INL', 'ILM', 'IPL']
        self.model = torchmodel
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        
        self.dataset = dataset
        self.oct_file = oct_file
        if self.dataset == 'bonn':
            self.oct_reader(self.oct_file)
            self.oct_metadata()
            self.get_bscan_fovea_bonn()
        self.mode = mode
    def __len__(self):
        return len(self.oct)

    def oct_reader(self, oct_file):
        self.oct = ep.import_heyex_vol(oct_file)

    def oct_metadata(self):
        self.scale_y = self.oct.meta.as_dict()['scale_y']
        self.scale_x = self.oct.meta.as_dict()['scale_x']
        self.visit_date = self.oct.meta.as_dict()['visit_date']
        self.laterality = self.oct.meta.as_dict()['laterality']
        self.loc_fovea = self.oct.meta.as_dict()['bscan_meta'][self.__len__() // 2]['start_pos'][1] // self.scale_x
        self.fovea_xstart = self.oct.meta.as_dict()['bscan_meta'][self.__len__() // 2]['start_pos'][0] // self.scale_x
        self.fovea_xstop = self.oct.meta.as_dict()['bscan_meta'][self.__len__() // 2]['end_pos'][0] // self.scale_x

    def print_metadata(self):
        print(len(self.oct))
        print('Y-Fovea: ', self.loc_fovea)
        print('ScaleY: ', self.scale_y)
        print('ScaleX: ', self.scale_x)
        print('Visit Date: ', self.visit_date)
        print('Laterality: ', self.laterality)

    def get_bscan_fovea_bonn(self):
        self.bscan_fovea = self.oct[len(self.oct)//2].data
        # bscans_start = len(self.oct) // 2 - 1
        # bscans_stop = len(self.oct) // 2 + 1

        # if len(self.oct) > 2:
        #     bs = []
        #     for idx in range(bscans_start, bscans_stop+1):   
        #         registration = np.array(self.oct[idx].data, dtype='float')
        #         bs.append(registration)
        #     self.bscan_fovea = np.array(bs, dtype='uint8').mean(axis=0)
        # else:
        #     self.bscan_fovea = self.oct[len(self.oct)//2].data
        x = 1
        
    def bscan_metadata(self, idx_bscan):
         self.bscan_start_pos = self.oct.meta.as_dict()['bscan_meta'][idx_bscan]['start_pos'][0]
         self.bscan_end_pos = self.oct.meta.as_dict()['bscan_meta'][idx_bscan]['end_pos'][0]

    def predict(self, model, x_image):
        transforms = T.Compose([
                    T.Normalize(mean=0.1338, std=0.1466) # CONTROL: 0.0389,  0.1036,  # FULL: 0.1338, 0.1466
        ])
        image = np.expand_dims(x_image, axis=-1)
        # image = np.repeat(image, 3, axis=-1)

        image = transforms(image=image)
        image = image['image'].transpose((2, 0, 1))
        image = torch.tensor(image, dtype=torch.float, device=self.device)
        image = image.unsqueeze(0)

        y_pred = model(image)
        y_pred = F.softmax(y_pred, dim=1)
        y_pred = torch.argmax(y_pred, dim=1)
        y_pred = y_pred.squeeze(0).detach().cpu().numpy()

        return y_pred

    def get_segmentation_large(self, img, gamma, alpha, imgw, imgh):
        img = gray_gamma(img, gamma=gamma)
        img = tv_denoising(img, alpha=alpha)

        img_h, img_w = img.shape
        
        image_x = F.interpolate(torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float(), (imgh, imgw), mode='bilinear', align_corners=False).squeeze().numpy()
        pred = self.predict(self.model, image_x)
        # print(pred.shape)
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

    def get_segmentation_patches(self, img, gamma, alpha, imgzs=128):
        img = gray_gamma(img, gamma=gamma)
        img = tv_denoising(img, alpha=alpha)
        pady = 0
        padx = 0
        if img.shape[0] == 496:
            pady = 8
        if img.shape[1] == 1000:
            padx = 12
        large_image = np.pad(img, [(pady, ), (padx, )], 'constant', constant_values=0)

        patches_images = patchify(large_image, (imgzs, imgzs), step=imgzs)

        preds = []
        for i in range(patches_images.shape[0]):
            for j in range(patches_images.shape[1]):
                image_x = patches_images[i, j, :, :]
                pred = self.predict(self.model, image_x)
                pred = Image.fromarray(pred.astype('uint8'))
                preds.append(np.array(pred))

        preds = np.reshape(preds, patches_images.shape)
        preds = np.array(preds)

        rec_img = unpatchify(patches=patches_images, imsize=(imgzs * preds.shape[0], imgzs * preds.shape[1]))

        rec_pred = unpatchify(patches=preds, imsize=(imgzs * preds.shape[0], imgzs * preds.shape[1]))

        shape_1 = (rec_pred.shape[0], rec_pred.shape[1], 3)

        rec_pred_rgb = np.zeros(shape=shape_1, dtype='uint8')

        norm = matplotlib.colors.Normalize(vmin=0, vmax=rec_pred.max())

        for idx in range(1, int(preds.max())+1):
            rec_pred_rgb[..., 0] = np.where(rec_pred == idx, cm.hsv(norm(idx), bytes=True)[0], rec_pred_rgb[..., 0])
            rec_pred_rgb[..., 1] = np.where(rec_pred == idx, cm.hsv(norm(idx), bytes=True)[1], rec_pred_rgb[..., 1])
            rec_pred_rgb[..., 2] = np.where(rec_pred == idx, cm.hsv(norm(idx), bytes=True)[2], rec_pred_rgb[..., 2])

        pred_overlay = Image.fromarray(rec_pred_rgb[pady:img.shape[0]+pady, padx:img.shape[1]+padx, :])
        img_overlay = Image.fromarray((rec_img[pady:img.shape[0]+pady, padx:img.shape[1]+padx]))
        pred_overlay = pred_overlay.convert("RGBA")
        img_overlay = img_overlay.convert("RGBA")

        overlay = Image.blend(img_overlay, pred_overlay, 0.5)

        return rec_pred[pady:img.shape[0]+pady, padx:img.shape[1]+padx], rec_pred_rgb[pady:img.shape[0]+pady, padx:img.shape[1]+padx, :], np.array(overlay)

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
        mask_opl = np.where(msk == self.classes.index('OPL'), 1, 0)
        pos_opl = np.where(mask_opl)
        ymin = np.min(pos_opl[0][np.nonzero(pos_opl[0])])

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

    def get_max_peak(self, img, window_size=1):
        max1 = []
        int_prof_x = []
        size = window_size
        k = 0
        for i in range(size, img.shape[1], size):
            indices = (-img[:, i - size:i].reshape(-1)).argsort()[:1]
            row1 = (int)(indices[0] / size)
            col1 = indices[0] - (row1 * size)
            temp1 = row1, col1 + k
            k += size
            max1.append(temp1)
            window = img[:, i - size:i]
            matrix_mean = np.zeros((img.shape[0]))
            for j in range(window.shape[0]):
                matrix_mean[j] = window[j, :].mean()
            int_prof_x.append(matrix_mean)
        max1 = np.array(max1)
        try:
            x1 = max1[:, 1]
        except:
            x1 = 0
        try:
            y1 = max1[:, 0]
        except:
            y1 = 0
        y1 = [float('nan') if x==0 else x for x in y1]
        return np.array(x1), np.array(y1)

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

    def get_individual_layers_segmentation(self, EZ_offset=4):
        self.binary_opl = self.get_layer_binary_mask(self.sample_pred, layer='OPL', offset=0)
        self.binary_elm = self.get_layer_binary_mask(self.sample_pred, layer='ELM', offset=0)
        self.binary_ez = self.get_layer_binary_mask(self.sample_pred, layer='EZ', offset=EZ_offset)
        self.binary_bm = self.get_layer_binary_mask(self.sample_pred, layer='BM', offset=0)
        
        self.segmented_opl = np.multiply(self.binary_opl, self.sample_bscan)
        self.segmented_elm = np.multiply(self.binary_elm, self.sample_bscan)
        self.segmented_ez = np.multiply(self.binary_ez, self.sample_bscan)
        self.segmented_bm = np.multiply(self.binary_bm, self.sample_bscan)

    def get_thickness(self, binary_image): 
        size = 1
        thickness = []
        for i in range(size, binary_image.shape[1], size):
            col = binary_image[:, i - size:i]
            if 1 in col:
                thickness.append(np.max(np.where(col)[0]) * self.scale_y - np.min(np.where(col)[0]) * self.scale_y)
        thickness_nan = np.array(thickness).copy()
        thickness_nan[thickness_nan == 0] = np.nan
        thickness_mean = np.nanmean(thickness_nan)
        thickness_std = np.nanstd(thickness_nan)
        return thickness_mean, thickness_std
    
    def get_area(self, binary_image):
        area_pixels = np.count_nonzero(binary_image == 1)
        return area_pixels

    def get_distance_in_mm(self, ref1, ref2): 
        distance_in_mm = []
        for i in range(ref1.shape[0]):
            distance_in_mm.append(np.abs(ref1[i] * self.scale_y - ref2[i] * self.scale_y))
        try:
            distance_in_mm_mean = np.nanmean(np.array(distance_in_mm))
            distance_in_mm_std = np.nanstd(np.array(distance_in_mm))
        except:
            distance_in_mm_mean = 0
            distance_in_mm_std = 0
        return np.array(distance_in_mm), distance_in_mm_mean, distance_in_mm_std

    def get_rEZI(self, ref1, ref2):
        rezi = []
        for i in range(ref1.shape[0]):
            relative_diff = ((ref1[i] - ref2[i]) / ((ref1[i] + ref2[i]) / 2)) * 100
            rezi.append(relative_diff)
        try:
            rezi_mean = np.nanmean(np.array(rezi))
            rezi_std = np.nanstd(np.array(rezi))
        except:
            rezi_mean = 0
            rezi_std = 0
        return np.array(rezi), rezi_mean, rezi_std

    def get_total_variation(self, segmentation, beta):
        y1 = segmentation
        vari = 0.0
        local = 0.0
        locales = []
        for k in range(0, y1.shape[1], 2 * beta):
            sample = y1[:, k:k + 2 * beta]
            for j in range(sample.shape[1]):
                vari = np.abs(sample[1, j] - sample[0, j])
                for i in range(2, sample.shape[0]):
                    dif = np.abs(sample[i, j] - sample[i - 1, j])
                    vari += dif
                local = vari / sample.shape[0]
            locales.append(local)
        locales = np.array(locales)
        locales_nan = locales.copy()
        locales_nan[locales_nan == 0] = np.nan
        return locales, np.nanmean(locales_nan), np.nanstd(locales_nan)

    def plot_overlay_oct_segmentation(self):
        fig, ax = plt.subplots(nrows=1, ncols=1, dpi=200, figsize=(14,10), gridspec_kw={'width_ratios': [1]}, frameon=False)
        ax.set_xlabel('B-Scan (X)', fontsize=14, weight="bold")
        ax.set_ylabel('A-Scan (Y)', fontsize=14, weight="bold")
        ax.tick_params(labelsize=12)
        ax.tick_params(labelsize=12)
        ax.imshow(self.overlay)

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

    def plot_segmentation_full(self):
        figure, ax = plt.subplots(nrows=1, ncols=1, figsize=(14,8), dpi=200, frameon=False)
        ax.imshow(self.overlay, cmap='gray')
        ax.set_xticks([])
        figure.tight_layout()

    def plot_segmentation_localization(self):
        figure, ax = plt.subplots(nrows=3, ncols=1, figsize=(14,8), gridspec_kw={'height_ratios': [2, 1, 1]}, dpi=200, frameon=False)
        ax[0].imshow(self.overlay, cmap='gray')
        ax[1].imshow(self.sample_overlay, cmap='gray')
        ax[1].set_xticks([])
        ax[2].imshow(self.segmented_total, cmap='gray')
        x1, y1 = [self.references[0], self.references[0]], [self.references[2], self.references[3]]
        x2, y2 = [self.references[1], self.references[1]], [self.references[2], self.references[3]]
        x3, y3 = [self.references[0], self.references[1]], [self.references[3], self.references[3]]
        x4, y4 = [self.references[0], self.references[1]], [self.references[2], self.references[2]]
        ax[0].plot(x1, y1, linewidth=2, color='y')
        ax[0].plot(x2, y2, linewidth=2, color='y')
        ax[0].plot(x3, y3, linewidth=2, color='y')
        ax[0].plot(x4, y4, linewidth=2, color='y')
        ax[0].set_xlabel('B-Scan (X)', fontsize=14, weight="bold")
        ax[0].set_ylabel('A-Scan (Y)', fontsize=14, weight="bold")
        ax[2].tick_params(labelsize=12)
        ax[0].tick_params(labelsize=12)
        ax[1].tick_params(labelsize=12)
        ax[2].tick_params(labelsize=12)
        figure.tight_layout()

    def plot_results(self):
        figure, ax = plt.subplots(nrows=3, ncols=1, figsize=(12,8), frameon=False, dpi=200)
        ax[0].imshow(self.sample_bscan, cmap='gray')
        ax[1].imshow(self.sample_overlay, cmap='gray')
        ax[2].imshow(self.sample_bscan, cmap='gray')
        ax[2].plot(self.max_opl_x, self.max_opl_y, c='cyan', label='Max Peaks OPL')
        ax[2].plot(self.max_ez_x, self.max_ez_y, linewidth=1.5,c='lime', label='Max Peaks EZ')
        ax[2].plot(self.lim_elm, c='violet', label='Limit ELM')
        ax[2].plot(self.lim_bm, c='red', label='Limit BM')
        ax[2].legend(loc='best')
        ax[0].set_ylabel('A-Scan (Y)', fontsize=14, weight="bold")
        ax[1].set_ylabel('A-Scan (Y)', fontsize=14, weight="bold")
        ax[2].set_xlabel('B-Scan (X)', fontsize=14, weight="bold")
        ax[2].set_ylabel('A-Scan (Y)', fontsize=14, weight="bold")
        ax[0].set_xticks([])
        ax[1].set_xticks([])
        ax[0].tick_params(labelsize=12)
        ax[1].tick_params(labelsize=12)
        ax[2].tick_params(labelsize=12)
        ax[2].tick_params(labelsize=12)
        figure.tight_layout()

    def plot_total_variation_alphas(self, alphas=[0.001, 0.005, 0.01, 0.05, 0.1], beta=3):
        p = np.percentile(self.bscan_fovea[self.references[2]:self.references[3], self.references[0]:self.references[1]], 95)
        sample_bscan1 = self.bscan_fovea[self.references[2]:self.references[3], self.references[0]:self.references[1]] / p

        EZ_segmented1 = np.multiply(self.binary_ez, sample_bscan1)
        locales, mean_tv, _ = self.get_total_variation(EZ_segmented1, beta)

        plt.figure(figsize=(14,6), dpi=128, frameon=False)
        plt.scatter(np.arange(locales.shape[0]), locales, s=5)
        plt.plot(locales, label=f'Original' + ' TV: ' + format(mean_tv,'.1e'))
        for w in alphas:
            tv_denoised = denoise_tv_chambolle(sample_bscan1, weight=w)
            EZ_segmented1 = np.multiply(self.binary_ez, tv_denoised)
            locales, mean_tv, _ = self.get_total_variation(EZ_segmented1, beta)
            plt.plot(locales, '--', linewidth=1, label=r'$\alpha$: ' +  format(w,'.1e') + ' TV: '+format(mean_tv,'.1e'))
        plt.xlabel(r'N/(2$\beta$+1)', fontsize=14, weight="bold")
        plt.ylabel('LV', fontsize=14, weight="bold")
        plt.legend(loc='right')
        figure.tight_layout()

    def plot_intensity_profiles(self, shift=1000):
        p = np.percentile(self.bscan_fovea[self.references[2]:self.references[3], self.references[0]:self.references[1]], 95)
        sample_bscan1 = self.bscan_fovea[self.references[2]:self.references[3], self.references[0]:self.references[1]] / p
        segmented_total1 = np.multiply(self.binary_total, sample_bscan1)
        img = segmented_total1
        int_prof_x = []
        size = 1

        for i in range(size, img.shape[1], size):
            window = img[:, i - size:i]
            matrix_mean = np.zeros((img.shape[0]))
            for j in range(window.shape[0]):
                matrix_mean[j] = window[j, :].mean()
            int_prof_x.append(matrix_mean)

        for t in range(240, np.array(int_prof_x).shape[0], shift):
            intensity = np.array(int_prof_x)[t, :]
            peaks, _ = find_peaks(intensity, height=0)
            y = np.arange(img.shape[0])
            fig, ax = plt.subplots(nrows=1, ncols=1)
            fig.set_figheight(4)
            fig.set_figwidth(12)
            ax.plot(intensity, y, 'k')
            ax.plot(intensity[peaks], peaks, "o", c='green')
            ax.set_xlabel('Grey value', fontsize=14, weight="bold")
            ax.set_ylabel(f'A-Scan \nDistance [Pixels]', fontsize=14, weight="bold")
            plt.gca().invert_yaxis()

        Z_INT = np.array(int_prof_x)
        x_int = np.arange(Z_INT.shape[1])
        y_int = np.arange(Z_INT.shape[0])
        X_INT, Y_INT = np.meshgrid(x_int, y_int)
        fig = plt.figure(figsize=(14, 8), dpi=200, frameon=False)
        ax = plt.axes(projection='3d')
        ax.set_aspect(aspect='auto', adjustable='datalim')
        ax.contour3D(X_INT, Y_INT, Z_INT, 100, cmap='jet')
        ax.set_xlabel('A-Scan(Y)', fontsize=14, weight="bold")
        ax.set_ylabel('B-Scan(X)', fontsize=14, weight="bold")
        ax.set_zlabel('Grey value', fontsize=14, weight="bold")
        ax.view_init(35, -45)      

    def fovea_forward(self, gamma, alpha, imgh, imgw, EZ_offset=3):
        self.pixel_2_mm2 = self.scale_x * self.scale_y

        if self.mode == 'large':
            self.pred_class_map, self.pred_rgb, self.overlay = self.get_segmentation_large(self.bscan_fovea, gamma=gamma, alpha=alpha, imgh=imgh, imgw=imgw)
        elif self.mode == 'patches':
            self.pred_class_map, self.pred_rgb, self.overlay = self.get_segmentation_patches(self.bscan_fovea, gamma=gamma, alpha=alpha, imgzs=imgh)
        else:
            raise RuntimeError('No mode delcared')
        ymin, ymax, xmin, xmax = self.get_reference_EZ_segmentation(msk=self.pred_class_map, offset=2)
        self.references = [xmin, xmax, ymin, ymax]

        self.sample_bscan = self.bscan_fovea[ymin:ymax,xmin:xmax]
        # p = np.percentile(self.sample_bscan, 95)
        # self.sample_bscan = self.sample_bscan / p
        self.sample_pred = self.pred_class_map[ymin:ymax,xmin:xmax]
        self.sample_pred_rgb = self.pred_rgb[ymin:ymax,xmin:xmax]
        self.sample_overlay = self.overlay[ymin:ymax,xmin:xmax]

        self.get_individual_layers_segmentation(EZ_offset=EZ_offset)
        
        self.max_opl_x, self.max_opl_y = self.get_max_peak(self.segmented_opl)
        self.max_ez_x, self.max_ez_y = self.get_max_peak(self.segmented_ez)
        self.lim_elm = self.get_limit(self.binary_elm, side='min')
        self.lim_bm = self.get_limit(self.binary_bm, side='max')

        self.ez_thickness_mean, self.ez_thickness_std = self.get_thickness(self.binary_ez) # EZ THICKNESS
        self.opl_thickness_mean, self.opl_thickness_std = self.get_thickness(self.binary_opl) # OPL THICKNESS
        self.elm_thickness_mean, self.elm_thickness_std = self.get_thickness(self.binary_elm) # ELM THICKNESS
        self.izrpe_thickness_mean, self.izrpe_thickness_std = self.get_thickness(self.binary_bm) # IZ+RPE THICKNESS

        self.ez_area_pixels = self.get_area(self.binary_ez) * self.scale_y * self.scale_x

        self.opl_2_elm, self.opl_2_elm_mean, self.opl_2_elm_std = self.get_distance_in_mm(self.max_opl_y, self.lim_elm) # DISTANCE FROM OPL TO ELM PEAKS IN MM
        self.opl_2_ez, self.opl_2_ez_mean, self.opl_2_ez_std = self.get_distance_in_mm(self.max_opl_y, self.max_ez_y) # DISTANCE FROM OPL TO EZ PEAKS IN MM
        self.elm_2_ez, self.elm_2_ez_mean, self.elm_2_ez_std = self.get_distance_in_mm(self.lim_bm, self.max_ez_y) # DISTANCE FROM ELM TO EZ PEAKS IN MM
        self.ez_2_bm, self.ez_2_bm_mean, self.ez_2_bm_std = self.get_distance_in_mm(self.max_ez_y, self.lim_bm) # DISTANCE FROM EZ TO BM PEAKS IN MM
        self.ez_total_variation, self.ez_total_variation_mean, self.ez_total_variation_std = self.get_total_variation(self.segmented_ez, 3) # TOTAL VARIATION
        self.rezi, self.rezi_mean, self.rezi_std = self.get_rEZI(self.max_ez_y, self.max_opl_y) # RELATIVE EZ INSTENSITY
        
        self.ez_fovea_limit = np.abs(xmax - xmin) * self.scale_x
        self.binary_total = np.where(np.logical_and(self.sample_pred.astype('float') <= 4, self.sample_pred.astype('float') > 0), 1, 0)
        self.segmented_total = np.multiply(self.binary_total, self.sample_bscan)

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
                if self.mode == 'large':
                    pred_class_map, _, _ = self.get_segmentation_large(bscan, gamma=2, alpha=0.03, imgh=imgh, imgw=imgw)
                elif self.mode == 'patches':
                    pred_class_map, _, _ = self.get_segmentation_patches(bscan, gamma=2, alpha=0.03, imgzs=imgh)
                else:
                    raise RuntimeError('No mode delcared')
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
