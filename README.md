# README

The goal in this project is to automatically segment the road from front camera view.

# Introduction<a name="intro"></a>

The goal of this project is to segment a drivable areas from a set of road images. These segmentations are used for autonomous vehicles. It provided the dataset that required much more focus on dealing with inconsistencies and differences in the limited data collection. As a result, most of our efforts went to trying out different ways to preprocess and combine different kinds of data.

### Demo

<img src="./figures/demo.gif" width="375"/>

### The Data

The data set consists of 8555 images. The data set consists of 8555 images, 4278 of which are cfc60 data and 4277 are cfc120 data.	


<table>
    <tr>
        <th>CFC</th>
        <th>CFCU</th>
    </tr>
    <tr>
        <th><img src='./figures/cfc.png' alt='CFC'></th>
        <th><img src='./figures/cfcu.jpg' alt='CFCU'></th>
    </tr>
</table>
 

### Workflow

![./figures/workflow.png](./figures/workflow.png)

# Preprocessing<a name="prep"></a>

### Json to Mask

Supervisely Annotation Format supports to polygon figures.

```json
{
    "id": 503004154,
    "classId": 1693021,
    "labelerLogin": "alexxx",
    "createdAt": "2020-08-21T15:15:28.092Z",
    "updatedAt": "2020-08-21T16:06:11.461Z",
    "description": "",
    "geometryType": "polygon",
    "tags": [],
    "classTitle": "Freespace",
    "points": {
        "exterior": [
            [
                730,
                2104
            ],
            [
                2479,
                402
            ],
            [
                3746,
                1646
            ]
        ],
        "interior": [
        [
            [
                1907,
                1255
            ],
            [
                2468,
                875
            ],
            [
                2679,
                1577
            ]
        ]
        ]
    }
}
```

- **geometryType**: "polygon" - class shape
- **exterior** - list of points [point1, point2, point3, etc ...] where each point is a list of two numbers (coordinates) [col, row]
- **interior** - list of elements with the same structure as the "exterior" field. In other words, this is the list of polygons that define object holes. For polygons without holes in them, this field is empty

The cv2 library's fillPoly function is used to draw a mask. Then, if the mask contains interior points, we exclude those points.

```python
mask = cv2.fillPoly(mask,np.array([obj['points']['exterior']]),color=1)
if len(obj['points']['interior']) != 0:
	mask = cv2.fillPoly(mask,inter,color=0)
```

<table>
    <tr>
        <th>With Hole</th>
        <th>Without Hole</th>
    </tr>
    <tr>
        <th><img src='./figures/hole.png' alt='hole'></th>
        <th><img src='./figures/nohole.png' alt='nohole'></th>
    </tr>
</table>

## One Hot Encode
This is where the integer encoded variable is removed and one new binary variable is added for each unique integer value in the variable. In the project we have two categories: freespace and background. Insert the value [0,1] for the background and [1,0] for the freespace.

### Sample

```python
encoded_labels = [[0,1],[1,0]]
        
        for i in range(n_classes):
            bl_mat = mask[:,:] == i
            encoded_data[bl_mat] = encoded_labels[i]
        
        return encoded_data
```

```matlab
[[1 0 0]                     [[[1 0][0 1][0 1]]
 [0 1 0]       ----->         [[0 1][1 0][0 1]]
 [1 0 0]]                     [[1 0][0 1][0 1]]]

```

## Augmentation

To avoid overfitting, I deployed data augmentation. 

<table>
    <tr>
        <th></th>
        <th>Image</th>
	<th></th>
        <th>Image</th>
    </tr>
    <tr>
        <td>Original Image</td>
        <td><img src='./figures/original.jpg' alt='Original'></td>
        <td>RandomBrightness</td>
        <td><img src='./figures/brightness.png' alt='RandomBrightness'></td>
    </tr>
    <tr>
        <td>RandomContrast</td>
	<td><img src='./figures/contrast.png' alt='RandomContrast'></td>
        <td>Gaussian Blur</td>
        <td><img src='./figures/blur.png' alt='Gaussian Blur'></td>
    </tr>
    <tr>
        <td>Crop</td>
        <td><img src='./figures/crop.png' alt='Crop'></td>
        <td>Horizontal Flip</td>
        <td><img src='./figures/horflip.png' alt='Horizontal Flip'></td>
    </tr>
</table>

---

<table>
    <tr>
        <th>Before Augmentation</th>
        <th>After Augmentation</th>
    </tr>
    <tr>
        <td><img src='./figures/before_366.png' alt='Before366'></td>
	<td><img src='./figures/after_366.jpg' alt='After366'></td>
    </tr>
    <tr>
        <td><img src='./figures/before_2449.png' alt='Before2449'></td>
	<td><img src='./figures/after_2449.jpg' alt='After2449'></td>
    </tr>
    <tr>
        <td><img src='./figures/before_3358.png' alt='Before3358'></td>
	<td><img src='./figures/after_3358.jpg' alt='After3358'></td>
    </tr>
</table>

# Cross Validation

Cross-validation is a resampling procedure used to evaluate machine learning models on a limited data sample.

- The dataset is divided into k groups.
- The selected group is used as the validation set.
- All other groups (k-1 groups) are used as train sets.

![./figures/split.png](./figures/split.png)

# Training

### Network architecture

Typical U-Net architecture was used in the project, which took in 2D image arrays with two channels (background,freespace).Then the inputs went through a series of convolutional and pooling layers and were turned into feature maps with smaller size. The resulting feature maps then pass a series of up-convolutional and concatenating layers and finally, the network output a segmentation mask.

![./figures/u-net-architecture.png](./figures/u-net-architecture.png)  

### Model Comparison

<img src="./figures/loss.png" width="375"/> <img src="./figures/iou.png" width="375"/> 

### Optimizer

The Adam optimizer was used to train the parameters.

![./figures/model.png](./figures/model.png)

## Predict Examples

<table>
    <tr>
        <th>Raw</th>
        <th>Labeled</th>
	<th>Prediction</th>
        <th>IoU Score</th>
    </tr>
    <tr>
	<td><img src='./figures/raw/446922_cfc_000183.jpg' alt='000183_raw'></td>
        <td><img src='./figures/labeled/446922_cfc_000183.png' alt='000183_labeled'></td>
	<td><img src='./figures/predicted/446922_cfc_000183.png' alt='000183_predicted'></td>
	<td>0.922321498</td>
    </tr>
    <tr>
	<td><img src='./figures/raw/446934_cfc_000195.jpg' alt='000195_raw'></td>
        <td><img src='./figures/labeled/446934_cfc_000195.png' alt='000195_labeled'></td>
	<td><img src='./figures/predicted/446934_cfc_000195.png' alt='000195_predicted'></td>
	<td>0.92544651</td>
    </tr>
    <tr>
	<td><img src='./figures/raw/446939_cfc_000200.jpg' alt='000200_raw'></td>
        <td><img src='./figures/labeled/446939_cfc_000200.png' alt='000200_labeled'></td>
	<td><img src='./figures/predicted/446939_cfc_000200.png' alt='000200_predicted'></td>
	<td>0.920089364</td>
    </tr>
    <tr>
	<td><img src='./figures/raw/446947_cfc_000208.jpg' alt='000208_raw'></td>
        <td><img src='./figures/labeled/446947_cfc_000208.png' alt='000208_labeled'></td>
	<td><img src='./figures/predicted/446947_cfc_000208.png' alt='000208_predicted'></td>
	<td>0.899553597</td>
    </tr>
    <tr>
	<td><img src='./figures/raw/446974_cfc_000235.jpg' alt='000235_raw'></td>
        <td><img src='./figures/labeled/446974_cfc_000235.png' alt='000235_labeled'></td>
	<td><img src='./figures/predicted/446974_cfc_000235.png' alt='000235_predicted'></td>
	<td>0.936607182</td>
    </tr>
    <tr>
	<td><img src='./figures/raw/446994_cfc_000255.jpg' alt='000255_raw'></td>
        <td><img src='./figures/labeled/446994_cfc_000255.png' alt='000255_labeled'></td>
	<td><img src='./figures/predicted/446994_cfc_000255.png' alt='000255_predicted'></td>
	<td>0.927678585</td>
    </tr>
    <tr>
	<td><img src='./figures/raw/447035_cfc_000296.jpg' alt='000296_raw'></td>
        <td><img src='./figures/labeled/447035_cfc_000296.png' alt='000296_labeled'></td>
	<td><img src='./figures/predicted/447035_cfc_000296.png' alt='000296_predicted'></td>
	<td>0.915178597</td>
    </tr>
    <tr>
	<td><img src='./figures/raw/447051_cfc_000312.jpg' alt='000312_raw'></td>
        <td><img src='./figures/labeled/447051_cfc_000312.png' alt='000312_labeled'></td>
	<td><img src='./figures/predicted/447051_cfc_000312.png' alt='000312_predicted'></td>
	<td>0.910267889</td>
    </tr>
    <tr>
	<td><img src='./figures/raw/447055_cfc_000316.jpg' alt='000316_raw'></td>
        <td><img src='./figures/labeled/447055_cfc_000316.png' alt='000316_labeled'></td>
	<td><img src='./figures/predicted/447055_cfc_000316.png' alt='000316_predicted'></td>
	<td>0.918303668</td>
    </tr>
    <tr>
	<td><img src='./figures/raw/447056_cfc_000317.jpg' alt='raw'></td>
        <td><img src='./figures/labeled/447056_cfc_000317.png' alt='labeled'></td>
	<td><img src='./figures/predicted/447056_cfc_000317.png' alt='predicted'></td>
	<td>0.907142937</td>
    </tr>
    <tr>
	<td><img src='./figures/raw/447134_cfc_000395.jpg' alt='raw'></td>
        <td><img src='./figures/labeled/447134_cfc_000395.png' alt='labeled'></td>
	<td><img src='./figures/predicted/447134_cfc_000395.png' alt='predicted'></td>
	<td>0.894642889</td>
    </tr>
    <tr>
	<td><img src='./figures/raw/447160_cfc_000421.jpg' alt='raw'></td>
        <td><img src='./figures/labeled/447160_cfc_000421.png' alt='labeled'></td>
	<td><img src='./figures/predicted/447160_cfc_000421.png' alt='predicted'></td>
	<td>0.917410731</td>
    </tr>
    <tr>
	<td><img src='./figures/raw/447196_cfc_000457.jpg' alt='raw'></td>
        <td><img src='./figures/labeled/447196_cfc_000457.png' alt='labeled'></td>
	<td><img src='./figures/predicted/447196_cfc_000457.png' alt='predicted'></td>
	<td>0.900446534</td>
    </tr>
    <tr>
	<td><img src='./figures/raw/447483_cfc_000744.jpg' alt='raw'></td>
        <td><img src='./figures/labeled/447483_cfc_000744.png' alt='labeled'></td>
	<td><img src='./figures/predicted/447483_cfc_000744.png' alt='predicted'></td>
	<td>0.872321486</td>
    </tr>
    <tr>
	<td><img src='./figures/raw/447522_cfc_000783.jpg' alt='raw'></td>
        <td><img src='./figures/labeled/447522_cfc_000783.png' alt='labeled'></td>
	<td><img src='./figures/predicted/447522_cfc_000783.png' alt='predicted'></td>
	<td>0.90982151</td>
    </tr>
    <tr>
	<td><img src='./figures/raw/447585_cfc_000846.jpg' alt='raw'></td>
        <td><img src='./figures/labeled/447585_cfc_000846.png' alt='labeled'></td>
	<td><img src='./figures/predicted/447585_cfc_000846.png' alt='predicted'></td>
	<td>0.916517854</td>
    </tr>
    <tr>
	<td><img src='./figures/raw/447593_cfc_000854.jpg' alt='raw'></td>
        <td><img src='./figures/labeled/447593_cfc_000854.png' alt='labeled'></td>
	<td><img src='./figures/predicted/447593_cfc_000854.png' alt='predicted'></td>
	<td>0.904910803</td>
    </tr>
    <tr>
	<td><img src='./figures/raw/447599_cfc_000860.jpg' alt='raw'></td>
        <td><img src='./figures/labeled/447599_cfc_000860.png' alt='labeled'></td>
	<td><img src='./figures/predicted/447599_cfc_000860.png' alt='predicted'></td>
	<td>0.906696498</td>
    </tr>
    <tr>
	<td><img src='./figures/raw/447692_cfc_000953.jpg' alt='raw'></td>
        <td><img src='./figures/labeled/447692_cfc_000953.png' alt='labeled'></td>
	<td><img src='./figures/predicted/447692_cfc_000953.png' alt='predicted'></td>
	<td>0.901785731</td>
    </tr>
    <tr>
	<td><img src='./figures/raw/447918_cfc_001179.jpg' alt='raw'></td>
        <td><img src='./figures/labeled/447918_cfc_001179.png' alt='labeled'></td>
	<td><img src='./figures/predicted/447918_cfc_001179.png' alt='predicted'></td>
	<td>0.92187506</td>
    </tr>
    <tr>
	<td><img src='./figures/raw/448020_cfc_001281.jpg' alt='raw'></td>
        <td><img src='./figures/labeled/448020_cfc_001281.png' alt='labeled'></td>
	<td><img src='./figures/predicted/448020_cfc_001281.png' alt='predicted'></td>
	<td>0.880803645</td>
    </tr>
    <tr>
	<td><img src='./figures/raw/449572_cfc_002833.jpg' alt='raw'></td>
        <td><img src='./figures/labeled/449572_cfc_002833.png' alt='labeled'></td>
	<td><img src='./figures/predicted/449572_cfc_002833.png' alt='predicted'></td>
	<td>0.917410731</td>
    </tr>
    <tr>
	<td><img src='./figures/raw/449709_cfc_002970.jpg' alt='raw'></td>
        <td><img src='./figures/labeled/449709_cfc_002970.png' alt='labeled'></td>
	<td><img src='./figures/predicted/449709_cfc_002970.png' alt='predicted'></td>
	<td>0.875446498</td>
    </tr>
    <tr>
	<td><img src='./figures/raw/449710_cfc_002971.jpg' alt='raw'></td>
        <td><img src='./figures/labeled/449710_cfc_002971.png' alt='labeled'></td>
	<td><img src='./figures/predicted/449710_cfc_002971.png' alt='predicted'></td>
	<td>0.874107182</td>
    </tr>
    <tr>
	<td><img src='./figures/raw/451568_cfc_004829.jpg' alt='raw'></td>
        <td><img src='./figures/labeled/451568_cfc_004829.png' alt='labeled'></td>
	<td><img src='./figures/predicted/451568_cfc_004829.png' alt='predicted'></td>
	<td>0.89419651</td>
    </tr>
    <tr>
	<td><img src='./figures/raw/451666_cfc_004927.jpg' alt='raw'></td>
        <td><img src='./figures/labeled/451666_cfc_004927.png' alt='labeled'></td>
	<td><img src='./figures/predicted/451666_cfc_004927.png' alt='predicted'></td>
	<td>0.905803561</td>
    </tr>
</table>