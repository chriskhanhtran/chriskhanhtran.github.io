---
title: "Visual Recognition for Vietnamese Foods"
date: 2020-05-26
categories:
  - posts
tags:
  - cv
  - deep learning
toc: true
toc_label: "Table of Contents"
toc_icon: "bookmark"
excerpt: "We are going to build a world-class image classifier using the fastai library to classify 11 popular Vietnamese dishes."
header:
  teaser: "https://www.google.com/logos/doodles/2020/celebrating-banh-mi-6753651837108330.3-2xa.gif"
---

[![](https://img.shields.io/badge/GitHub-View_Repository-blue?logo=GitHub)](https://github.com/chriskhanhtran/vn-food-app)

Imagine that you are a world traveller and are traveling to a country famous for its street foods. Walking in a night market street full of food trucks with many delicious-looking options, you have no idea what these foods are and whether they contain any ingredient you are allergic to. You want to ask the local but you don't know the language. You wish that you have an app on your phone that allows you to take a picture of that food you want to have and will return all the information you need to know about it.

It's one simple application of computer vision that can make our life better, besides many other applications being implemented in autonomous driving or cancer detection.

Today we are going to build a world-class image classifier using the `fastai` library to classify 11 popular Vietnamese dishes. The `fastai` library is built on top of [PyTorch](https://pytorch.org/) and allows us to quickly and easily build the latest neural networks and train our models to receive state-of-the-art results.

Before we begin, let's load packages that we are going to use in this project.


```python
%reload_ext autoreload
%autoreload 2
%matplotlib inline
```


```python
import os
import numpy as np
import urllib.request
from fastai.vision import *
from fastai.metrics import error_rate
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
```

## Looking at the data

I have built an image dataset of popular Vietnamese dishes using the Bing Image Search API by following [PyImageSearch's tutorial](https://www.pyimagesearch.com/2018/04/09/how-to-quickly-build-a-deep-learning-image-dataset/). We can directly download this dataset from Google Drive.


```python
%%time
# Download and unzip
if not os.path.exists("data"):
    !wget -O "dataset.zip" "https://www.googleapis.com/drive/v3/files/13GD8pcwHJPiAPbPtm6KeC20Qw1zm9xdy?alt=media&key=AIzaSyCmo6sAQ37OK8DK4wnT94PoLx5lx-7VTDE"
    !unzip dataset.zip
    !rm dataset.zip
```


Each class is stored in a seperate folder in `data` and we can use `ImageDataBunch.from_folder` to quickly load our dataset. In addition, we resize our images to 224 x 224 pixel and use `get_transforms` to flip, rotate, zoom, warp, adjust lighting our original images (which is called data augmentation, a strategy that enables practitioners to significantly increase the diversity of data available for training models, without actually collecting new data). We also `normalize` the images using statistics from ImageNet dataset.


```python
path = Path("./data/")
np.random.seed(42)
data = ImageDataBunch.from_folder(path, train='.', ds_tfms=get_transforms(),
                                  valid_pct=0.2, bs=64, size=224).normalize(imagenet_stats)
```

The dataset has more than 6k images and we use 20% of them as a validation set.


```python
len(data.train_ds), len(data.valid_ds)
```




    (4959, 1239)



Let's look at some samples:


```python
data.show_batch(rows=3, figsize=(7, 6))
```


![png](/assets/images/vn-food-classifier/output_12_0.png)


The dataset has a total of 11 classes. They are:


```python
print("Number of classes: ", len(data.classes))
print("Classes: ", data.classes)
```

    Number of classes:  11
    Classes:  ['banh-mi', 'banh-xeo', 'bubble-tea', 'bun-bo-hue', 'bun-bo-nam-bo', 'bun-cha', 'bun-dau-mam-tom', 'che', 'hu-tieu', 'pho', 'spring-rolls']


## Training: ResNet-50

Now we will finetune a ResNet-50 model on our customized dataset. ResNet is from the paper [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf) and is the best default model for computer vision. This ResNet-34 model is trained on ImageNet with 1000 classes, so first we need to initialize a new head for the model to be adapted to the number of classes in our dataset. The `cnn_learner` method will do this for us (Read [documentation](https://docs.fast.ai/vision.learner.html#cnn_learner)).

Then we will train the model in two stages:
- first we freeze the body weights and only train the head,
- then we unfreeze the layers of the backbone and fine-tune the whole model.

### Stage 1: Finetune the top layers only


```python
learn = cnn_learner(data, models.resnet50, metrics=error_rate, pretrained=True)
```

We will train the model for 20 epoches following the 1cycle policy. The number of epoches we need to train depends on how different the dataset is from ImageNet. Basically, we train the model until the validation loss stops decrease. The default learning rate here is 3e-3.


```python
learn.fit_one_cycle(20)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.356109</td>
      <td>0.657256</td>
      <td>0.206618</td>
      <td>01:31</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.870719</td>
      <td>0.560793</td>
      <td>0.167070</td>
      <td>01:27</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.687549</td>
      <td>0.526645</td>
      <td>0.167877</td>
      <td>01:32</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.621711</td>
      <td>0.444908</td>
      <td>0.143664</td>
      <td>01:27</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.493342</td>
      <td>0.397156</td>
      <td>0.126715</td>
      <td>01:28</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.435479</td>
      <td>0.381771</td>
      <td>0.123487</td>
      <td>01:27</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.373893</td>
      <td>0.389900</td>
      <td>0.121065</td>
      <td>01:28</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.325564</td>
      <td>0.371386</td>
      <td>0.112994</td>
      <td>01:28</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.295842</td>
      <td>0.349679</td>
      <td>0.106538</td>
      <td>01:29</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.271255</td>
      <td>0.348150</td>
      <td>0.098467</td>
      <td>01:28</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.233799</td>
      <td>0.317944</td>
      <td>0.091203</td>
      <td>01:28</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.205257</td>
      <td>0.306772</td>
      <td>0.086360</td>
      <td>01:27</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.171645</td>
      <td>0.310830</td>
      <td>0.084746</td>
      <td>01:28</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.159442</td>
      <td>0.293081</td>
      <td>0.078289</td>
      <td>01:27</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.137362</td>
      <td>0.295972</td>
      <td>0.083939</td>
      <td>01:29</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.112108</td>
      <td>0.289382</td>
      <td>0.084746</td>
      <td>01:26</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.093431</td>
      <td>0.279121</td>
      <td>0.083939</td>
      <td>01:30</td>
    </tr>
    <tr>
      <td>17</td>
      <td>0.094814</td>
      <td>0.281516</td>
      <td>0.079903</td>
      <td>01:29</td>
    </tr>
    <tr>
      <td>18</td>
      <td>0.083980</td>
      <td>0.277226</td>
      <td>0.073446</td>
      <td>01:33</td>
    </tr>
    <tr>
      <td>19</td>
      <td>0.090189</td>
      <td>0.274643</td>
      <td>0.081517</td>
      <td>01:28</td>
    </tr>
  </tbody>
</table>



```python
learn.save('resnes50-stage-1')
```

### Stage 2: Unfreeze and finetune the entire networks

At this stage we will unfreeze the whole model and finetune it with smaller learning rates. The code below will help us find the learning rate for this stage.


```python
learn.load('resnes50-stage-1')
learn.lr_find()
learn.recorder.plot()
```

![png](/assets/images/vn-food-classifier/output_24_2.png)


When finetuning the model at state 2, we will use different learning rates for different layers. The top layers will be update at greater rates than the bottom layers. As a rule of thumb, we use learning rates between (a, b), in which:
- a is taken from the LR Finder above where the loss starts to decrease for a while,
- b is 5 to 10 times smaller than the default rate we used in stage 1.

In this case, the learning rate we will use is from 3e-6 to 3e-4 We continnue to train the model until the validation loss stops decreasing.


```python
learn.unfreeze()
learn.fit_one_cycle(8, max_lr=slice(3e-6, 3e-4))
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.077156</td>
      <td>0.277789</td>
      <td>0.079903</td>
      <td>01:30</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.089096</td>
      <td>0.269736</td>
      <td>0.075061</td>
      <td>01:30</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.092252</td>
      <td>0.282279</td>
      <td>0.079096</td>
      <td>01:29</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.074603</td>
      <td>0.265167</td>
      <td>0.067797</td>
      <td>01:31</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.071408</td>
      <td>0.278237</td>
      <td>0.074253</td>
      <td>01:33</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.047503</td>
      <td>0.250248</td>
      <td>0.065375</td>
      <td>01:29</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.037788</td>
      <td>0.249852</td>
      <td>0.070218</td>
      <td>01:30</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.035603</td>
      <td>0.251429</td>
      <td>0.066990</td>
      <td>01:29</td>
    </tr>
  </tbody>
</table>


After two stages, the error rate on the validation set is about 6% and the accuracy for our dataset with 11 classes is 94%. That's a pretty accurate model. It's a good practice to save the current stage of the model in case we want to train it for more epoches.


```python
learn.save('resnes50-stage-2')
```

## Evaluation

Now let's look on the model's predictions. Specifically, we will look at examples where the model makes wrong predictions. Below are 9 examples with the top losses, meaning on these examples the model gives the actual class a very low score.


```python
learn.load('resnes50-stage-2')
interp = ClassificationInterpretation.from_learner(learn)
losses, idxs = interp.top_losses()
interp.plot_top_losses(9, figsize=(15,11))
```

![png](/assets/images/vn-food-classifier/output_31_1.png)


The confusion matrix below will us a big picture of which pair of dishes make our model confused. The number of wrong predictions are quite low.


```python
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
```


![png](/assets/images/vn-food-classifier/output_33_0.png)


```python
interp.most_confused(min_val=3)
```




    [('pho', 'hu-tieu', 11),
     ('bun-bo-nam-bo', 'bun-cha', 5),
     ('hu-tieu', 'bun-bo-nam-bo', 5),
     ('pho', 'bun-bo-hue', 5),
     ('bun-bo-hue', 'pho', 4),
     ('hu-tieu', 'bun-cha', 4),
     ('bun-bo-hue', 'hu-tieu', 3),
     ('bun-cha', 'bun-bo-nam-bo', 3),
     ('hu-tieu', 'bun-bo-hue', 3),
     ('hu-tieu', 'pho', 3)]



## Production

After being satisfied with our model, we can save it to disk and use it to classify new data. We can also deploy it into a magical app that I described at the beginning of the notebook.


```python
# Export model
learn.export()
```

For inference, we also want to run the model on CPU instead of GPU.


```python
defaults.device = torch.device('cpu')
```

The functions below will download an image from an URL and use the model to make prediction for that image.


```python
def open_image_url(url):
    """Download image from URL, return fastai image and PIL image."""
    urllib.request.urlretrieve(url, "./img/test.jpg")
    return open_image("./img/test.jpg"), Image.open("./img/test.jpg")

def predict(url):
    """Make prediction for image from url, show image and predicted probability."""
    img, pil_img = open_image_url(url)
    pred_class, pred_idx, outputs = learn.predict(img)
    print("Predicted Class: ", pred_class)
    print(f"Probability: {outputs[pred_idx].numpy() * 100:.2f}%")
    
    plt.figure(figsize=(15, 4))
    # Show image
    plt.subplot(1, 2, 1)
    plt.imshow(pil_img)
    plt.axis('off')
    plt.title('Image')
    
    # Plot Probabilities
    plt.subplot(1, 2, 2)
    pd.Series(outputs, data.classes).sort_values().plot(kind='barh')
    plt.xlabel("Class")
    plt.title("Probability");
```


```python
# Load model
learn = load_learner(path)
```


```python
# Make prediction from URL
url = str(input("URL: "))
predict(url)
```

    URL:  https://i.pinimg.com/originals/9b/63/d7/9b63d76a44be020e03eeaec0a1134e95.jpg


    Predicted Class:  bun-bo-hue
    Probability: 99.99%



![png](/assets/images/vn-food-classifier/output_43_2.png)


I have also built a web app for this model. Check it out below:

[![](https://img.shields.io/badge/Heroku-Open_Web_App-blue?logo=Heroku)](https://vietnamese-food.herokuapp.com/)

![](https://github.com/chriskhanhtran/vn-food-app/blob/master/img/vn-food-app.gif?raw=true)

## Conclusion

We have walked through the whole pipeline to finetune a ResNet-50 model on our customized dataset, evaluate its strengths and weaknesses and deploy the model for production. With the `fastai` library, we can quickly achieve state-of-the-art results with very neat and clean codes.

In this project, we built a classifier for only 11 popular Vietnamese dishes, but we can easily scale up the model by collecting more data for hundreds or thounsands of classes following this [PyImageSearch's tutorial](https://www.pyimagesearch.com/2018/04/09/how-to-quickly-build-a-deep-learning-image-dataset/).

I have seen many interesting ideas using computer vision, such as [Plantix](https://play.google.com/store/apps/details?id=com.peat.GartenBank&hl=en_US), a plant doctor app which can tell what diseases your plants might have from a single photo and how you should take care of them. And yes, with some software engineering skills, some data and an idea, you can build your own impactful computer vision application. I look forwards to seeing these applications make the world a better place in all directions.

## Reference
- [Fast AI: Lesson 1 - What's your pet](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson1-pets.ipynb)
- [Fast AI: Lesson 2 - Creating your own dataset from Google Images](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson2-download.ipynb)
- [PyImageSearch: How to (quickly) build a deep learning image dataset](https://www.pyimagesearch.com/2018/04/09/how-to-quickly-build-a-deep-learning-image-dataset/)
