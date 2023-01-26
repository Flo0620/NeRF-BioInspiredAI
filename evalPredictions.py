import numpy as np
import os
import cv2
from skimage.metrics import structural_similarity as ssim
# from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
import math
import torch
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

loss_fn_alex = lpips.LPIPS(net='vgg')

img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))

def imgToLpipsInput(img):
    img=np.array([[img[:,:,0],img[:,:,1],img[:,:,2]]])

    img = img-img.min()
    img=((img/img.max())*2)-1
    img = torch.from_numpy(img)
    img = img.to(torch.float32)
    return img

def imgToPsnrlnInput(img):
    img=(img/img.max())
    img = torch.from_numpy(img)
    img = img.to(torch.float32)
    return img


def sampleNPixelsToTest(imgT,imgP,N_rand):
    H,W=imgT.shape[0],imgT.shape[1]
    coords = np.stack(np.meshgrid(np.linspace(0, H-1, H), np.linspace(0, W-1, W)), -1)
    coords = np.reshape(coords, [-1,2])  # (H * W, 2)
    select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
    select_coords = coords[select_inds].astype('int32')  # (N_rand, 2)
    imgT=imgT[select_coords[:,0],select_coords[:,1]]
    imgP=imgP[select_coords[:,0],select_coords[:,1]]
    imgT=np.reshape(imgT,(int(math.sqrt(N_rand)),int(math.sqrt(N_rand)),3))
    imgP=np.reshape(imgP,(int(math.sqrt(N_rand)),int(math.sqrt(N_rand)),3))

    
    return imgT,imgP


#directory="/home/florian/nerf-pytorch2/nerf-pytorch/logs/cello_white/TestRun1/testset_200000"
#directoryTest="/home/florian/Schreibtisch/BAI Dataset/newDatasets/cello_white/test"
#directory="/home/florian/nerf-pytorch2/nerf-pytorch/logs/legoRunFullResTry3/FullRes500k/testset_500000"
#directoryTest="/home/florian/nerf-pytorch2/nerf-pytorch/data/nerf_synthetic/lego/test"
#directory="/home/florian/Schreibtisch/LegoTest148Generated"
#directoryBlack="/home/florian/Schreibtisch/LegoTest148Black"
#directoryTest="/home/florian/Schreibtisch/LegoTest148Blender"
# directory="/home/florian/nerf-pytorch2/nerf-pytorch/logs/fern200k/fern_test/testset_200000"
#directory="/home/florian/testset_100000"
#directory="/home/florian/nerf-pytorch2/nerf-pytorch/logs/fern/Run2Factor4/testset_200000"
#directoryTest="/home/florian/nerf-pytorch2/nerf-pytorch/data/nerf_llff_data/fern/images_4"
# directory = "/home/florian/BAIDatasetTrainingResults/rubiks_cube/testset_200000"
# directoryTest="/home/florian/Schreibtisch/BAI Dataset/newDatasets/rubiks_cube/test"
directory = "/home/florian/BAIDatasetTrainingResults/house/testset_200000"
directoryTest="/home/florian/Schreibtisch/BAI Dataset/newDatasets/cello/test"
directory="/home/florian/testset_020000"

#number of pixels that should be sampled for the calculation of the PSNR
N_rand = 1024

imgL = []
# psnrL = []
psnrlnL = []
ssimL = []
lpipsL = []

for img in os.listdir(directory):
    imgNameTest = str((int)(img.split(".")[0]))+".png" #own datasets
    #imgNameTest = "r_"+str((int)(img.split(".")[0]))+".png" #lego
    #imgNameTest = "image"+str((img.split(".")[0]))+".png" #fern
    imgPred=cv2.imread(directory+"/"+img)
    imgPred=cv2.cvtColor(imgPred,cv2.COLOR_BGR2RGB)
    imgTest=cv2.imread(directoryTest+"/"+imgNameTest,cv2.IMREAD_UNCHANGED)


    testimg_alpha_channel_vorhanden=imgTest.shape[2]==4

    if(testimg_alpha_channel_vorhanden == True):
        #Put white background behind transparent regions as it is done in the original paper
        imgTestMult = np.stack((imgTest[:,:,3]/255,imgTest[:,:,3]/255,imgTest[:,:,3]/255),-1)
        imgTestAdd = np.stack(((255-imgTest[:,:,3]),(255-imgTest[:,:,3]),(255-imgTest[:,:,3])),-1)
        imgTest=imgTest[:,:,0:3]
        imgTest = imgTest*imgTestMult+imgTestAdd
        imgTest = imgTest.astype('uint8')
    imgTest=cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

    imgTestPsnr,imgPredPsnr=sampleNPixelsToTest(imgTest,imgPred,N_rand)

    ssimScore = ssim(imgTest,imgPred,channel_axis=2)
    print("ssimScore "+img+": "+str(ssimScore))

    imgPredPsnrln=imgToPsnrlnInput(imgPredPsnr)
    imgTestPsnrln=imgToPsnrlnInput(imgTestPsnr)

    psnrlnScore = mse2psnr(img2mse(imgTestPsnrln,imgPredPsnrln))
    print("psnrlnScore: "+str(psnrlnScore.item()))


    imgPred=imgToLpipsInput(imgPred)
    imgTest=imgToLpipsInput(imgTest)
    lpipsScore=loss_fn_alex(imgTest,imgPred)
    print("lpipsScore "+img+": "+str(lpipsScore[0][0][0][0].item()))

    imgL.append(img)
    psnrlnL.append(psnrlnScore.item())
    # psnrL.append(psnrScore)
    ssimL.append(ssimScore)
    lpipsL.append(lpipsScore[0][0][0][0].item())

imgL = np.array(imgL)
psnrlnL = np.array(psnrlnL)
# psnrL = np.array(psnrL)
ssimL= np.array(ssimL)
lpipsL = np.array(lpipsL)

print("---PSNR---")
print("mean: "+str(np.mean(psnrlnL)))
print("min: img "+str(imgL[np.argmin(psnrlnL)])+": "+str(np.min(psnrlnL)))
print("max: img "+str(imgL[np.argmax(psnrlnL)])+": "+str(np.max(psnrlnL)))
print("---SSIM---")
print("mean: "+str(np.mean(ssimL)))
print("min: img "+str(imgL[np.argmin(ssimL)])+": "+str(np.min(ssimL)))
print("max: img "+str(imgL[np.argmax(ssimL)])+": "+str(np.max(ssimL)))
print("---LPIPS---")
print("mean: "+str(np.mean(lpipsL)))
print("min: img "+str(imgL[np.argmin(lpipsL)])+": "+str(np.min(lpipsL)))
print("max: img "+str(imgL[np.argmax(lpipsL)])+": "+str(np.max(lpipsL)))