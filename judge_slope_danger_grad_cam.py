"""
20200130
Predict ID from Image by Pytorch
Fine Tunning
Grad-cam
"""
import itertools

import numpy as np

from tqdm import tqdm
import os

import torch
from torchvision import models
from torchvision import datasets
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
from torch import nn, optim
import random

import cv2
import torch.nn.functional as F
#import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns
import  count_pic

#fix the RNG
seed = 0
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

 

def all():
    class GradCam:
        def __init__(self, model, feature_layer, device):
            self.model = model
            self.feature_layer = feature_layer
            self.device = device
            self.model.eval()
            self.feature_grad = None
            self.feature_map = None
            self.hooks = []

            # save the gradient of last layer in CNN. in resnet layer = module
            #  layer = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            def save_feature_grad(module, in_grad, out_grad):
                self.feature_grad = out_grad[0]
            self.hooks.append(self.feature_layer.register_backward_hook(save_feature_grad))

            # get the feature of last layer in CNN
            def save_feature_map(module, inp, outp):
                self.feature_map = outp[0]
            self.hooks.append(self.feature_layer.register_forward_hook(save_feature_map))

        def forward(self, x):
            return self.model(x)

        def backward_on_target(self, output, target):
            self.model.zero_grad()
            one_hot_output = torch.zeros([1, output.size()[-1]])# output.size()[-1] = class num
            one_hot_output[0][target] = 1
            one_hot_output = one_hot_output.to(self.device) # self.device
            output.backward(gradient=one_hot_output, retain_graph=True)

        def clear_hook(self):
            for hook in self.hooks:
                hook.remove()


    #*****************************************************************************
    #STEP1 Make DataLoader  ******************************************************
    class ImageFolderWithPaths(datasets.ImageFolder):
        """Custom dataset that includes image file paths.
        Extends torchvision.datasets.ImageFolder
        """
        # override the __getitem__ method. this is the method that dataloader calls
        def __getitem__(self, index):
            # this is what ImageFolder normally returns
            original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
            # the image file path
            path = self.imgs[index][0]
            # make a new tuple that includes original and the path
            tuple_with_path = (original_tuple + (path,))
            return tuple_with_path


    def make_loader(mypath_train, mypath_predict, train_batch_size, train_ratio):
        print("=== STEP1 Make DataLoader ===")
        #Make Dataset
        size = 224
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        train_val_imgs = ImageFolderWithPaths(mypath_train,
                                              transform = transforms.Compose([
                                                                             transforms.Resize(size),
                                                                             transforms.RandomHorizontalFlip(),
                                                                             transforms.CenterCrop(size),
                                                                             transforms.ToTensor(),
                                                                             transforms.Normalize(mean, std)
                                                                             ])
                                              )
        n_imgs = len(train_val_imgs)
        train_size = int(n_imgs * train_ratio)
        val_size = n_imgs - train_size
        train_imgs, val_imgs =  torch.utils.data.random_split(train_val_imgs, [train_size, val_size])

    #    pred_imgs = ImageFolderWithPaths(mypath_predict,
    #                                     transform = transforms.Compose([
    #                                                                     transforms.ToTensor(),
    #                                                                     transforms.Normalize(mean, std)
    #                                                                     ])
    #                                     )

        pred_imgs = ImageFolderWithPaths(mypath_predict,
                                         transform = transforms.Compose([
                                                                         transforms.ToTensor(),
                                                                         transforms.Normalize(mean, std)
                                                                         ])
                                         )


        print(" train_imgs={}, val_imgs={}, pred_imgs={}".format(len(train_imgs),len(val_imgs),len(pred_imgs)))

        #Make DataLoader
        train_loader = DataLoader(train_imgs, batch_size=train_batch_size, shuffle=True)
        val_loader = DataLoader(val_imgs, batch_size=train_batch_size, shuffle=True)
        pred_loader = DataLoader(pred_imgs, batch_size=1, shuffle=False)

        return train_loader, val_loader, pred_loader


    #******************************************************************************
    #STEP2 Set NetWork  ***********************************************************
    def set_network(class_num):
        print("\n=== STEP2 Set NetWork ===")
        net_res18 = models.resnet18(pretrained=True) #use the preTrained Parameters
        #Set update param
        #update_params_names_res18_1 = ["features"]
        #update_params_names_res18_2 = ["classifier.0.weight", "classifier.0.bias", "classifier.3.weight", "classifier.3.bias"]
        #update_params_names_res18_3 = ["classifier.6.weight", "classifier.6.bias"]
        #params_to_update_res18_1 =[]
        #params_to_update_res18_2 =[]
        #params_to_update_res18_3 =[]
        #
        #for name, param in net_res18.named_parameters():
        #    if update_params_names_res18_1[0] in name:
        #        param.requires_grad = True
        #        params_to_update_res18_1.append(param)
        #    elif name in update_params_names_res18_2:
        #        param.requires_grad = True
        #        params_to_update_res18_2.append(param)
        #    elif name in update_params_names_res18_3:
        #        param.requires_grad = True
        #        params_to_update_res18_3.append(param)
        #    else:
        #        param.requires_grad = False #fix the Trained W,B while Training

        #Change last layer output
        fc_input_dim = net_res18.fc.in_features
        net_res18.fc = nn.Linear(fc_input_dim, class_num)
        return net_res18


    #******************************************************************************
    #STEP3 Train the NetWork  *****************************************************
    def train_net(net, train_loader, val_loader, optimizer, loss_fn, n_iter, device):
        print("\n=== STEP3 Train the NetWork ===")
        epoch_losses_tra = []
        epoch_corrects_ratio_tra = []
        epoch_losses_val = []
        epoch_corrects_ratio_val = []

        for epoch in range(n_iter):
            epoch_loss_tra = 0.0
            epoch_correct_tra = 0
            n_tra = 0
            epoch_loss_val = 0.0
            epoch_correct_val = 0
            n_val = 0

            for i_tra, (x, y, _) in tqdm(enumerate(train_loader), total=len(train_loader)):
                net.train()
                x = x.to(device)
                y = y.to(device)

                ypre = net(x)
                _, ypreid = ypre.max(1)

                loss = loss_fn(ypre, y)

                if epoch > 0:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                epoch_loss_tra += loss.item()
                epoch_correct_tra += (ypreid == y).float().sum().item()
                n_tra += len(x)

            for i_val, (x, y, _) in tqdm(enumerate(val_loader), total=len(val_loader)):
                net.eval()
                x = x.to(device)
                y = y.to(device)

                ypre = net(x)
                _, ypreid = ypre.max(1)

                loss = loss_fn(ypre, y)

                epoch_loss_val += loss.item()
                epoch_correct_val += (ypreid == y).float().sum().item()
                n_val += len(x)

            epoch_losses_tra.append(epoch_loss_tra / (i_tra + 1))
            epoch_corrects_ratio_tra.append(epoch_correct_tra / n_tra)
            epoch_losses_val.append(epoch_loss_val / (i_val + 1))
            epoch_corrects_ratio_val.append(epoch_correct_val / n_val)

            print(" train  epoch={0}, epoch_losses={1:.3f}, epoch_corrects_ratio={2:.3f}".format(epoch, epoch_losses_tra[-1], epoch_corrects_ratio_tra[-1]), flush=True)
            print(" val    epoch={0}, epoch_losses={1:.3f}, epoch_corrects_ratio={2:.3f}".format(epoch, epoch_losses_val[-1], epoch_corrects_ratio_val[-1]), flush=True)


    #******************************************************************************
    #STEP4 Use Trained NetWork  ***************************************************
    def predict_grad_cam(net, pred_loader, device):
        fnames = []
        cams = []
        net.eval()
        # Last layer in CNN of Resnet is "net.layer4.modules())[-1]"
        #grad_cam = GradCam(net, feature_layer=list(net.layer4.modules())[-1], device=device)
        for x, y, fname in tqdm(pred_loader):
            # Last layer in CNN of Resnet is "net.layer4.modules())[-1
            grad_cam = GradCam(net, feature_layer=list(net.layer4.modules())[-1], device=device)
            fnames.append(fname)
            # x.size = batch, channels, height, widht
            x = x.to(device)

            x_size = (x.size()[3], x.size()[2])# widht, height

            # input image in GradCam
            grad_cam_output_val = grad_cam.forward(x)# = net output
            # get the class No
            target = grad_cam_output_val.argmax(1).item()

            #backward
            grad_cam.backward_on_target(grad_cam_output_val, target)
            # Get feature gradient (while backward)
            feature_grad = grad_cam.feature_grad.data.to("cpu").detach().numpy()[0].copy()
            # Get weights from gradient
            weights = np.mean(feature_grad, axis=(1, 2))  # Take averages for each gradient

            # Get features outputs
            feature_map = grad_cam.feature_map.data.to("cpu").numpy().copy()

            grad_cam.clear_hook()

            # Get cam
            cam = np.sum((weights * feature_map.T), axis=2).T
            cam = np.maximum(cam, 0)  # use np.maximum instead of ReLU

            cam = cv2.resize(cam, x_size)
            cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
            cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
    #        cam = cam.transpose(1, 0)
    #        cam =cam[np.newaxis, :, :]
            cams.append(cam)

        return fnames, cams


    def predict_grad_cam_by_trained_model(net, pred_loader, device):
        pre_out = predict_grad_cam(net, pred_loader, device)

        #Save the NetWork OutPut
        filenames = np.array(pre_out[0]) # list to numpy #filenames
        grad_cam_vals = pre_out[1] # GPU tensor to numpy #ypre_vals

        return filenames, grad_cam_vals


    def predict_net(net, pred_loader, device):
        net.eval() #invalid Dropout and Batchnorm
        fnames = []
        ypre_vals = []
        ypre_classes = []

        for x, y, fname in tqdm(pred_loader):
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():#Processing required for automatic differentiation is off
                ypre = net(x)
                ypre = nn.functional.softmax(ypre, dim=1)
                _, ypre_class = ypre.max(1)
            fnames.append(fname)
            ypre_vals.append(ypre)
            ypre_classes.append(ypre_class)

        ypre_vals = torch.cat(ypre_vals)
        ypre_classes = torch.cat(ypre_classes)

        return fnames, ypre_vals, ypre_classes


    def predict_by_trained_model(net, pred_loader, device):
        print("\n=== STEP4 Predict by Trained NetWork ===")
        #predict using Trained Model
        pre_out = predict_net(net, pred_loader, device)

        #Save the NetWork OutPut
        filenames = np.array(pre_out[0]) #list to numpy #filenames
        pred_vals = pre_out[1].cpu().numpy() #GPU tensor to numpy #ypre_vals
        pred_classes = pre_out[2].cpu().numpy() #ypre_classes

        return filenames, pred_vals, pred_classes


    def judge_slope(parent_path,
                    mypath_train,
                    mypath_predict,
                    n_iter,
                    train_batch_size,
                    class_num,
                    device,
                    train_ratio
                    ):

        parent_path = parent_path + "/"
        mypath_train = parent_path + mypath_train + "/"
        mypath_predict = parent_path + mypath_predict + "/"

        d_loader = make_loader(mypath_train, mypath_predict, train_batch_size, train_ratio)#Rturn train_loader, val_loader, pred_loader
        train_loader = d_loader[0]
        val_loader = d_loader[1]
        pred_loader = d_loader[2]

        net_res18 = set_network(class_num)
        net_res18.to(device)
        optimizer_res18 = optim.Adam(net_res18.parameters())


        if os.path.exists(parent_path+"judge_slople_danger.prm") == True:
           print("\n=== STEP3 Import trained model__judge_slople_danger.prm")
           net_res18.load_state_dict(torch.load(parent_path+"judge_slople_danger.prm",map_location='cpu'))
           net_res18.eval()
        else:
           print(" fine_tunning_ResNet18")
           train_net(net=net_res18,
                     train_loader=train_loader,
                     val_loader=val_loader,
                     optimizer=optimizer_res18,
                     loss_fn=nn.CrossEntropyLoss(),
                     n_iter=n_iter,
                     device=device
                     )

           trained_params = net_res18.state_dict()
           torch.save(trained_params, parent_path+"judge_slople_danger.prm")

        pred_out = predict_by_trained_model(net_res18, pred_loader, device)
        pfiles = pred_out[0]
        pvals = pred_out[1]
        pclasses = pred_out[2]

        # Grad Cam
        grad_cam_out = predict_grad_cam_by_trained_model(net_res18, pred_loader, device)
        gcam_files = grad_cam_out[0]
        gcam_vals = grad_cam_out[1]

        return pfiles, pvals, pclasses, gcam_files, gcam_vals
    #    return pfiles, pvals, pclasses

    """
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pred_out = judge_slope(parent_path="../images",
                           mypath_train="train_test_img",
                           mypath_predict="predict_img",
                           n_iter = 50,
                           train_batch_size = 32,
                           class_num = 2,
                           device = device,
                           train_ratio = 0.8
                           )
    #print("predict_images\n", pred_out[0])
    #print("\npredict_result\n", pred_out[1])
    #print("\npredict_class\n", pred_out[2])
    #print("\ngrad_cam_images\n", pred_out[3])#pred_out[0]と同じファイル名。確認用に出力してる。
    #print("\ngrad_cam_out\n", pred_out[4])

    # pred_out[0]は予測対象のファイル名
    # pred_out[1]は予測対象に対するネットワーク出力値
    # pred_out[2]はpred_out[1]の値に基づいて分類されるクラスの数値
    # pred_out[3]はpred_out[0]と同じファイル名。チェック用に出力。
    # pred_out[4]は予測対象の画像におけるgrad-cam出力値。AIが画像のドコを主に見ているかを把握する数値。
    #      3次元配列なので、例えば、squeezeで２次元配列にしてから、seabornライブラリで描画などしてビジュアル化する。


    """
    ここから下はサンプルなので、このpyファイルを外から呼び出すときはコメント行にしておいてください
    """
    Heatmap_l=[]
    for idd in range(pred_out[0].size):

        gcam_imgs = pred_out[4]
        gcam_imgs = gcam_imgs[idd]#1枚目の写真

        gcam_imgs_np = np.array(gcam_imgs)
        print(pred_out[3][idd])
        print(gcam_imgs_np.shape)
        del gcam_imgs
        #グラフA

        sns.heatmap(gcam_imgs_np, square=True)# seaborn は(縦、横)の配列で与える

        #グラフB(グラフAでもBでも、それ以外でもOK)
        #plt.savefig("../images/predict_img/original/heatmap{0}.jpg".format(idd))
        pred_out3_l2d=pred_out[3].tolist()
        pred_out3=list(itertools.chain.from_iterable(pred_out3_l2d))
        heatmap_name="Heatmap"+os.path.basename(pred_out3[idd])
        count_heatmap=count_pic.count_pic(path="./Heatmap")
        Heatmap_l.append(count_heatmap+heatmap_name)
        plt.savefig("Heatmap/"+count_heatmap+heatmap_name)
        plt.savefig("../images/predict_img/original/"+count_heatmap+heatmap_name)
        # plt.figure()
        plt.close('all')
        del pred_out3
        del pred_out3_l2d
    print("predict_images\n",pred_out[0])
    print("\npredict_result\n",pred_out[1])
    print("\npredict_class\n",pred_out[2])
    print("\ngrad_cam_images\n",pred_out[3])
    # print("\ngrad_cam_out\n",pred_out[4])
    return pred_out[0],pred_out[1],pred_out[2],pred_out[3],pred_out[4],Heatmap_l
if __name__ == "__main__":
    all()
# plt.figure()
# plt.imshow(gcam_imgs_np, interpolation="nearest")
# plt.colorbar()
# plt.s