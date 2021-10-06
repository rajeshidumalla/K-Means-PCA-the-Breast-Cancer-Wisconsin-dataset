# K-Means & PCA

### Setup

Let's setup Spark on Colab environment.  Run the cell below!


```python
!pip install pyspark
!pip install -U -q PyDrive
!apt install openjdk-8-jdk-headless -qq
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
```

    Requirement already satisfied: pyspark in /usr/local/lib/python3.7/dist-packages (3.1.2)
    Requirement already satisfied: py4j==0.10.9 in /usr/local/lib/python3.7/dist-packages (from pyspark) (0.10.9)
    openjdk-8-jdk-headless is already the newest version (8u292-b10-0ubuntu1~18.04).
    0 upgraded, 0 newly installed, 0 to remove and 37 not upgraded.


Now we import some of the libraries usually needed by our workload.

---








```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import pyspark
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark import SparkContext, SparkConf
```

Let's initialize the Spark context.


```python
# create the session
conf = SparkConf().set("spark.ui.port", "4050")

# create the context
sc = pyspark.SparkContext(conf=conf)
spark = SparkSession.builder.getOrCreate()
```

I can easily check the current version and get the link of the web interface. In the Spark UI, I can monitor the progress of my job and debug the performance bottlenecks (if my Colab is running with a **local runtime**).


```python
spark
```





    <div>
        <p><b>SparkSession - in-memory</b></p>

<div>
    <p><b>SparkContext</b></p>

    <p><a href="http://a1a933967aa2:4050">Spark UI</a></p>

    <dl>
      <dt>Version</dt>
        <dd><code>v3.1.2</code></dd>
      <dt>Master</dt>
        <dd><code>local[*]</code></dd>
      <dt>AppName</dt>
        <dd><code>pyspark-shell</code></dd>
    </dl>
</div>

    </div>




If I run this on the Google colab hosted runtime, the cell below will create a *ngrok* tunnel which will allow me to still check the Spark UI.


```python
!wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
!unzip ngrok-stable-linux-amd64.zip
get_ipython().system_raw('./ngrok http 4050 &')
!curl -s http://localhost:4040/api/tunnels | python3 -c \
    "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"
```

    --2021-10-06 00:43:26--  https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
    Resolving bin.equinox.io (bin.equinox.io)... 52.202.168.65, 54.237.133.81, 18.205.222.128, ...
    Connecting to bin.equinox.io (bin.equinox.io)|52.202.168.65|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 13832437 (13M) [application/octet-stream]
    Saving to: ‘ngrok-stable-linux-amd64.zip.2’
    
    ngrok-stable-linux- 100%[===================>]  13.19M  50.3MB/s    in 0.3s    
    
    2021-10-06 00:43:26 (50.3 MB/s) - ‘ngrok-stable-linux-amd64.zip.2’ saved [13832437/13832437]
    
    Archive:  ngrok-stable-linux-amd64.zip
    replace ngrok? [y]es, [n]o, [A]ll, [N]one, [r]ename: yes
      inflating: ngrok                   
    https://c7fb-35-190-173-94.ngrok.io


### Data Preprocessing

In this Notebook, rather than downloading a file from some where, I will load a famous machine learning dataset, the [Breast Cancer Wisconsin dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html), using the ```scikit-learn``` datasets loader.


```python
from sklearn.datasets import load_breast_cancer
breast_cancer = load_breast_cancer()
```

For convenience, given that the dataset is small, I will first construct a Pandas dataframe, tune the schema, and then convert it into a Spark dataframe.


```python
pd_df = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
pd_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean radius</th>
      <th>mean texture</th>
      <th>mean perimeter</th>
      <th>mean area</th>
      <th>mean smoothness</th>
      <th>mean compactness</th>
      <th>mean concavity</th>
      <th>mean concave points</th>
      <th>mean symmetry</th>
      <th>mean fractal dimension</th>
      <th>radius error</th>
      <th>texture error</th>
      <th>perimeter error</th>
      <th>area error</th>
      <th>smoothness error</th>
      <th>compactness error</th>
      <th>concavity error</th>
      <th>concave points error</th>
      <th>symmetry error</th>
      <th>fractal dimension error</th>
      <th>worst radius</th>
      <th>worst texture</th>
      <th>worst perimeter</th>
      <th>worst area</th>
      <th>worst smoothness</th>
      <th>worst compactness</th>
      <th>worst concavity</th>
      <th>worst concave points</th>
      <th>worst symmetry</th>
      <th>worst fractal dimension</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17.99</td>
      <td>10.38</td>
      <td>122.80</td>
      <td>1001.0</td>
      <td>0.11840</td>
      <td>0.27760</td>
      <td>0.3001</td>
      <td>0.14710</td>
      <td>0.2419</td>
      <td>0.07871</td>
      <td>1.0950</td>
      <td>0.9053</td>
      <td>8.589</td>
      <td>153.40</td>
      <td>0.006399</td>
      <td>0.04904</td>
      <td>0.05373</td>
      <td>0.01587</td>
      <td>0.03003</td>
      <td>0.006193</td>
      <td>25.38</td>
      <td>17.33</td>
      <td>184.60</td>
      <td>2019.0</td>
      <td>0.1622</td>
      <td>0.6656</td>
      <td>0.7119</td>
      <td>0.2654</td>
      <td>0.4601</td>
      <td>0.11890</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20.57</td>
      <td>17.77</td>
      <td>132.90</td>
      <td>1326.0</td>
      <td>0.08474</td>
      <td>0.07864</td>
      <td>0.0869</td>
      <td>0.07017</td>
      <td>0.1812</td>
      <td>0.05667</td>
      <td>0.5435</td>
      <td>0.7339</td>
      <td>3.398</td>
      <td>74.08</td>
      <td>0.005225</td>
      <td>0.01308</td>
      <td>0.01860</td>
      <td>0.01340</td>
      <td>0.01389</td>
      <td>0.003532</td>
      <td>24.99</td>
      <td>23.41</td>
      <td>158.80</td>
      <td>1956.0</td>
      <td>0.1238</td>
      <td>0.1866</td>
      <td>0.2416</td>
      <td>0.1860</td>
      <td>0.2750</td>
      <td>0.08902</td>
    </tr>
    <tr>
      <th>2</th>
      <td>19.69</td>
      <td>21.25</td>
      <td>130.00</td>
      <td>1203.0</td>
      <td>0.10960</td>
      <td>0.15990</td>
      <td>0.1974</td>
      <td>0.12790</td>
      <td>0.2069</td>
      <td>0.05999</td>
      <td>0.7456</td>
      <td>0.7869</td>
      <td>4.585</td>
      <td>94.03</td>
      <td>0.006150</td>
      <td>0.04006</td>
      <td>0.03832</td>
      <td>0.02058</td>
      <td>0.02250</td>
      <td>0.004571</td>
      <td>23.57</td>
      <td>25.53</td>
      <td>152.50</td>
      <td>1709.0</td>
      <td>0.1444</td>
      <td>0.4245</td>
      <td>0.4504</td>
      <td>0.2430</td>
      <td>0.3613</td>
      <td>0.08758</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11.42</td>
      <td>20.38</td>
      <td>77.58</td>
      <td>386.1</td>
      <td>0.14250</td>
      <td>0.28390</td>
      <td>0.2414</td>
      <td>0.10520</td>
      <td>0.2597</td>
      <td>0.09744</td>
      <td>0.4956</td>
      <td>1.1560</td>
      <td>3.445</td>
      <td>27.23</td>
      <td>0.009110</td>
      <td>0.07458</td>
      <td>0.05661</td>
      <td>0.01867</td>
      <td>0.05963</td>
      <td>0.009208</td>
      <td>14.91</td>
      <td>26.50</td>
      <td>98.87</td>
      <td>567.7</td>
      <td>0.2098</td>
      <td>0.8663</td>
      <td>0.6869</td>
      <td>0.2575</td>
      <td>0.6638</td>
      <td>0.17300</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20.29</td>
      <td>14.34</td>
      <td>135.10</td>
      <td>1297.0</td>
      <td>0.10030</td>
      <td>0.13280</td>
      <td>0.1980</td>
      <td>0.10430</td>
      <td>0.1809</td>
      <td>0.05883</td>
      <td>0.7572</td>
      <td>0.7813</td>
      <td>5.438</td>
      <td>94.44</td>
      <td>0.011490</td>
      <td>0.02461</td>
      <td>0.05688</td>
      <td>0.01885</td>
      <td>0.01756</td>
      <td>0.005115</td>
      <td>22.54</td>
      <td>16.67</td>
      <td>152.20</td>
      <td>1575.0</td>
      <td>0.1374</td>
      <td>0.2050</td>
      <td>0.4000</td>
      <td>0.1625</td>
      <td>0.2364</td>
      <td>0.07678</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = spark.createDataFrame(pd_df)

def set_df_columns_nullable(spark, df, column_list, nullable=False):
    for struct_field in df.schema:
        if struct_field.name in column_list:
            struct_field.nullable = nullable
    df_mod = spark.createDataFrame(df.rdd, df.schema)
    return df_mod

df = set_df_columns_nullable(spark, df, df.columns)
df = df.withColumn('features', array(df.columns))
vectors = df.rdd.map(lambda row: Vectors.dense(row.features))

df.printSchema()
```

    root
     |-- mean radius: double (nullable = false)
     |-- mean texture: double (nullable = false)
     |-- mean perimeter: double (nullable = false)
     |-- mean area: double (nullable = false)
     |-- mean smoothness: double (nullable = false)
     |-- mean compactness: double (nullable = false)
     |-- mean concavity: double (nullable = false)
     |-- mean concave points: double (nullable = false)
     |-- mean symmetry: double (nullable = false)
     |-- mean fractal dimension: double (nullable = false)
     |-- radius error: double (nullable = false)
     |-- texture error: double (nullable = false)
     |-- perimeter error: double (nullable = false)
     |-- area error: double (nullable = false)
     |-- smoothness error: double (nullable = false)
     |-- compactness error: double (nullable = false)
     |-- concavity error: double (nullable = false)
     |-- concave points error: double (nullable = false)
     |-- symmetry error: double (nullable = false)
     |-- fractal dimension error: double (nullable = false)
     |-- worst radius: double (nullable = false)
     |-- worst texture: double (nullable = false)
     |-- worst perimeter: double (nullable = false)
     |-- worst area: double (nullable = false)
     |-- worst smoothness: double (nullable = false)
     |-- worst compactness: double (nullable = false)
     |-- worst concavity: double (nullable = false)
     |-- worst concave points: double (nullable = false)
     |-- worst symmetry: double (nullable = false)
     |-- worst fractal dimension: double (nullable = false)
     |-- features: array (nullable = false)
     |    |-- element: double (containsNull = false)
    


With the next cell, I am going build the two datastructures that we will be using throughout this Notebook:


*   ```features```, a dataframe of Dense vectors, containing all the original features in the dataset;
*   ```labels```, a series of binary labels indicating if the corresponding set of features belongs to a subject with breast cancer, or not.



```python
from pyspark.ml.linalg import Vectors
features = spark.createDataFrame(vectors.map(Row), ["features"])
labels = pd.Series(breast_cancer.target)
```

### Building machine learning model

Now I am ready to cluster the data with the [K-means](https://spark.apache.org/docs/latest/ml-clustering.html) algorithm included in MLlib (Spark's Machine Learning library).
Also, I am setting  the ```k``` parameter to **2**, fit the model, and the compute the [Silhouette score](https://en.wikipedia.org/wiki/Silhouette_(clustering)) (i.e., a measure of quality of the obtained clustering).  

**IMPORTANT:** I am using the MLlib implementation of the Silhouette score (via ```ClusteringEvaluator```).


```python
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# Trains a k-means model.
kmeans = KMeans().setK(2).setSeed(1)
model = kmeans.fit(features)
```


```python
# Make predictions
predictions = model.transform(features)

# Evaluate clustering by computing Silhouette score
evaluator = ClusteringEvaluator()

silhouette = evaluator.evaluate(predictions)
```


```python
print(f'Silhouette: {silhouette}')
```

    Silhouette: 0.8342904262826145


Next, I will take the predictions produced by K-means, and compare them with the ```labels``` variable (i.e., the ground truth from our dataset).  

Then, I will compute how many data points in the dataset have been clustered correctly (i.e., positive cases in one cluster, negative cases in the other).

I am using ```np.count_nonzero(series_a == series_b)``` to quickly compute the element-wise comparison of two series.

**IMPORTANT**: K-means is a clustering algorithm, so it will not output a label for each data point, but just a cluster identifier!  As such, label ```0``` does not necessarily match the cluster identifier ```0```.



```python
predictions_df = predictions.toPandas()
converted_pre = predictions_df['prediction'].apply(lambda x: 0 if x else 1)
np.count_nonzero(converted_pre.values == labels.values)
```




    486



Now I am performing dimensionality reduction on the ```features``` using the [PCA](https://spark.apache.org/docs/latest/ml-features.html#pca) statistical procedure, available as well in MLlib.

Setting the ```k``` parameter to **2**, effectively reducing the dataset size of a **15X** factor.


```python
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors

pca = PCA(k=2, inputCol="features", outputCol="pca")
model = pca.fit(features)

result = model.transform(features).select("pca")
result.show(truncate=False)
```

    +-----------------------------------------+
    |pca                                      |
    +-----------------------------------------+
    |[-2260.0138862925405,-187.9603012226368] |
    |[-2368.993755782054,121.58742425815508]  |
    |[-2095.6652015478608,145.11398565870087] |
    |[-692.6905100570508,38.576922592081765]  |
    |[-2030.2124927427062,295.2979839927924]  |
    |[-888.280053576076,26.079796157025726]   |
    |[-1921.082212474845,58.807572473099206]  |
    |[-1074.7813350047961,31.771227808469668] |
    |[-908.5784781618829,63.83075279044624]   |
    |[-861.5784494075679,40.57073549705321]   |
    |[-1404.559130649947,88.23218257736237]   |
    |[-1524.2324408687816,-3.2630573167779793]|
    |[-1734.385647746415,273.1626781511459]   |
    |[-1162.9140032230355,217.63481808344613] |
    |[-903.4301030498832,135.61517666084782]  |
    |[-1155.8759954206848,76.80889383742165]  |
    |[-1335.7294321308068,-2.4684005450356024]|
    |[-1547.2640922523087,3.805675972574325]  |
    |[-2714.9647651812156,-164.37610864258804]|
    |[-908.2502671870876,118.216420082231]    |
    +-----------------------------------------+
    only showing top 20 rows
    


Now running K-means with the same parameters as above, but on the ```pcaFeatures``` produced by the PCA reduction that I just executed.

I am also computing the Silhouette score, as well as the number of data points that have been clustered correctly.


```python
kmeans = KMeans(featuresCol='pca').setK(2).setSeed(1)
model = kmeans.fit(result)

pca_predictions = model.transform(result)
pca_evaluator = ClusteringEvaluator(featuresCol='pca')

pca_silhouette = pca_evaluator.evaluate(pca_predictions)

print(f'Silhouette after PCS {pca_silhouette}')
```

    Silhouette after PCS 0.8348610363444836



```python
pca_predictions_df = pca_predictions.toPandas()
pca_converted_pre = pca_predictions_df['prediction'].apply(lambda x: 0 if x else 1)
np.count_nonzero(pca_converted_pre == labels.values)
```




    486




```python
#stopping Spark environment
sc.stop()
```
