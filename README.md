# StylePatch
the source code of the StylePatch（a adversarial patch attack method using the local style fusion）

### Introduction

------

 StylePatch is an approach to enhance the stealthiness of adversarial patches through a framework founded on local style fusion. This methodology commences with the identification of image areas susceptible to perturbation, accomplished through weighted class  activation mapping, facilitating precise localization of the optimal placement for adversarial patches. Employing style  transfer techniques, the framework computes style and content matrices between the target image and the adversarial patch.  During the adversarial patch generation process, the adversarial patch's style and content are refined using the cosine distance function to enable seamless fusion with the style and content of the local image it overlays. This integration into the  ambient environment results in color and style harmonization, thereby mitigating the patch's perceptibility to the human  visual system. The overarching objective is to amplify the covert nature of adversarial patches. Experimental findings affirm the utility of this approach, as it engenders adversarial patches characterized by both clandestine attributes and effectiveness, enabling surreptitious attacks imperceptible to human observers.

### Setup

------

Python 3.6.9

Ubuntu 18.04

2 * RTX 2080Ti 

all the libs are listed in the **requirements.txt** , you must pip it before running the code. 

### How to Run

After the setup stages，you can run the code in these steps:

1.First , get the dataset(imageNet 2012) at [ImageNet (image-net.org)](https://image-net.org/)

2.Then get the Pretrained Pytorch Model

3.running in the Terminal

```
python3 StylePatch_test.py
```
4.If you want to generate adversarial samples by stylepatch, upload your images to StylePatch/input/000, and you could change the trainsize in stylepatch.py.
This code will ensure the pixeles of output image are same as input image except the patch area because I've removed the normalized function. And you could control the pixel float accuracy variation in patch area.
This code will also ensure the name of output image is same as input image.
If you want to try the functions of the other three attack models or see the ASR（Attack Success Rate） of model, just merge the corresponding code of StylePatch_test.py into this stylepatch.py.
```
python3 stylepatch.py
```
### Result
![result](https://github.com/xie-1399/StylePatch/blob/main/pic/result.png)


