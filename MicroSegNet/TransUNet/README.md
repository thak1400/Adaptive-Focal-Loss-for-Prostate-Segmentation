# MicroSegNet
Official PyTorch implementation of: 

PROSTATE CANCER SEGMENTATION FROM MICRO-ULTRASOUND IMAGES USING ADAPTIVE FOCAL LOSS

## Requirements
* Python==3.10.6
* torch==2.1.0
* torchvision==0.16.0
* numpy
* opencv-python
* tqdm
* tensorboard
* tensorboardX
* ml-collections
* medpy
* SimpleITK
* scipy
* Execute the file: `pip install -r requirements.txt`

## Usage
### 1. Download Google pre-trained ViT models
* "R50+ViT-B_16.npz" is used here. [Get model in this link](https://storage.googleapis.com/vit_models/imagenet21k/R50%2BViT-B_16.npz). 
* Create a folder named "imagenet21k" under "vit_checkpoint" and place the model there, rename the model as: R50+ViT-B_16.npz
* Save your model into folder "model\vit_checkpoint\imagenet21k\".
* This is what the path will look like: "MicroSegNet\model\vit_checkpoint\imagenet21k\R50+ViT-B_16.npz"

### 2. Prepare data
* Please go to https://zenodo.org/records/10475293 to download the dataset.
* After downloading, extract the file and put it into folder "data/". The directory structure should be as follows:

```bash
.
├── data
│   ├── Micro_Ultrasound_Prostate_Segmentation_Dataset
│   │   ├── train
│   │	└── test
│   └── preprocessing.py
│
├── model
│   └── vit_checkpoint
│       └── imagenet21k
│           ├── R50+ViT-B_16.npz
└── TransUNet

```

* Run the preprocessing script, which would generate training images in folder "train_png/", data list files in folder "lists/" and data.csv for overview.
```
python preprocessing.py
```
* Training images are preprocessed to 224*224 to feed into networks.

### 3. Train/Test
* Please go to the folder "TransUNet/" and it's ready for you to train and test the model.
```
python train_MicroUS.py
python test_MicroUS.py
```
By default the model will be trained on the Adaptive Focal Loss function.

* If you with to train it on PyTorch's Focal Loss function then go to trainer_MicroUS.py and comment these lines: 
            
            loss0 = adaptive_focal_loss(label0_batch, out0, non_expert0_batch,1)
            loss1 = adaptive_focal_loss(label1_batch, out1, non_expert1_batch,2)
            loss2 = adaptive_focal_loss(label2_batch, out2, non_expert2_batch,3)
            loss3 = adaptive_focal_loss(label_batch, outputs, non_expert_batch,4)
            
            And uncomment these:
            # loss0 = pyT_focal_loss(label0_batch, out0)
            # loss1 = pyT_focal_loss(label1_batch, out1)
            # loss2 = pyT_focal_loss(label2_batch, out2)
            # loss3 = pyT_focal_loss(label_batch, outputs)

* If you with to train it on the default AG-BCE loss function then uncomment these lines and comment the other loss functions: 
            
            # loss0 = attention_BCE_loss(hard_weight, label0_batch, out0, non_expert0_batch,1)
            # loss1 = attention_BCE_loss(hard_weight, label1_batch, out1, non_expert1_batch,2)
            # loss2 = attention_BCE_loss(hard_weight, label2_batch, out2, non_expert2_batch,3)
            # loss3 = attention_BCE_loss(hard_weight, label_batch, outputs, non_expert_batch, 4)\

### 4. Gradio App:

* Run the training srcipt separately using all 3 loss functions, one at a time.

* Once the model is saved, go to MicroSegNet\model\ and change the name of model to "MicroSegNet_MicroUS224_R50-ViT-B_16_weight4_epo10_bs8_adaptiveFocal" for model trained on the Adaptive Focal Loss function,
"MicroSegNet_MicroUS224_R50-ViT-B_16_weight4_epo10_bs8_default" for the model trained on the AG-BCE function,
"MicroSegNet_MicroUS224_R50-ViT-B_16_weight4_epo10_bs8_pyTFocal" for the model trained on PyTorch's Focal Loss function.

* Run the "gradioApp" file in VSCode and open the link provided

* In the Input Image section selection the image(s) you want to segment the prostate for.

* Select the objective function by checking the corresponding check box. Multiple can also be selected.

* Click the Submit button and wait for the predictions to appear in the Contour Images section. You can view the images there.

* If you wish to download the image(s) then click on the hyperlink provided in the Download Contour Image section.

* Click clear to reset the app.


