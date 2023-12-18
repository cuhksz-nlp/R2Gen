import cv2
import os
import numpy as np

def fft2d(image):
    # 1) compute 1d-fft on columns
    fftcols = np.array([fft(row) for row in image]).transpose()

    # 2) next, compute 1d-fft on in the opposite direction (for each row) on the resulting values
    return np.array([fft(row) for row in fftcols]).transpose()

# change "root_folder" & "output_folder" before you run the program !!
root_folder = 'C:/Users/Windows 11/Desktop/iu_xray/images'

for subfolder in os.listdir(root_folder):
    patient_folder_path = root_folder + "/" + subfolder

    for image_file in os.listdir(patient_folder_path):
        full_image_file = patient_folder_path + "/" + image_file

        output_folder = 'C:/Users/Windows 11/Desktop/iu_xray_histogram/images'
        output_folder_path = output_folder + "/" + subfolder + "/" + image_file

        img = cv2.imread(full_image_file, 0)
        img_fft = np.fft.fftshift(np.fft.fft2(img))
        M,N = img_fft.shape
        H = np.zeros((M,N), dtype=np.float32)
        D0 = 10
        for u in range(M):
            for v in range(N):
                D = np.sqrt((u-M/2)**2 + (v-N/2)**2)
                H[u,v] = np.exp(-D**2/(2*D0*D0))
        FLP = H * img_fft
        FLP = np.fft.ifftshift(FLP)
        fLP = np.abs(np.fft.ifft2(FLP))
        gMask = img - fLP
        k = 5
        g = img + k * gMask
        g = np.clip(g, 0, 255)
        G = (1 + k*(1-H)) * img_fft
        g = np.abs(np.fft.ifft2(np.fft.ifftshift(G)))
        g = np.clip(g, 0, 255)
        # hpf = img - cv2.GaussianBlur(img, (21, 21), 3)+127
        # equ = cv2.equalizeHist(hpf)
        try:
            os.mkdir(output_folder + "/" + subfolder)
        except:
            pass
        # print(output_folder_path)
        print(output_folder_path)
        cv2.imwrite(output_folder_path, g)
