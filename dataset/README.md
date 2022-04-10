## 该文件夹是用来存放训练样本的目录

### 使用步骤：
* (1) `dataset`目录下创建`flower_data`子目录
* (2) 下载花分类数据集 [flower_photos.tgz](http://download.tensorflow.org/example_images/flower_photos.tgz)
* (3) 解压数据集 (`tar zxvf flower_photos.tgz  -C ./`)
* (4) 划分训练集和验证集 (执行`split_data.py`)  

```
├── flower_data   
       ├── flower_photos（解压的数据集文件夹，3670个样本）  
       ├── train（生成的训练集，3306个样本）  
       └── val（生成的验证集，364个样本） 
```