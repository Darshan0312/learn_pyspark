# 🚀 Master PySpark Cheat Sheet
*A complete guide covering data processing, MLlib, Spark NLP, deep learning integrations, and more*

---

## 🔹 Introduction
PySpark lets you use Python to work with big data using Apache Spark.  
It enables distributed data processing across clusters efficiently.  
Think of it like **pandas for big data** — but built for scale.

---

## ⚙️ Starting a Spark Session
```python
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("App").getOrCreate()
```

---

## 📥 Reading Data
```python
spark.read.csv("file.csv", header=True, inferSchema=True)
spark.read.json("file.json")
spark.read.parquet("file.parquet")
spark.read.option("multiLine", True).json("file.json")
spark.read.text("file.txt")
spark.read.format("jdbc").options(...).load()
```

---

## 📤 Writing Data
```python
df.write.csv("output.csv", header=True)
df.write.mode("overwrite").parquet("out.parquet")
```

---

## 🔍 Exploring Data
```python
df.show()
df.printSchema()
df.describe().show()
df.summary().show()
df.columns
df.dtypes
df.count()
```

---

## 🚫 Null Handling and Duplicates
```python
df.dropna()
df.fillna(0)
df.dropDuplicates()
df.dropDuplicates(["col1"])
```

---

## 🧪 Column Operations
```python
from pyspark.sql import functions as F

df.withColumn("new_col", df["col"] * 2)
df.withColumn("flag", F.when(df["col"] > 100, 1).otherwise(0))
df.withColumnRenamed("old", "new")
```

---

## 🎯 Filtering Data
```python
df.filter(df["col"] > 50)
df.where(df["status"] == "active")
df.filter(df["col"].isin("A", "B"))
```

---

## 📊 Aggregations
```python
df.groupBy("col").sum("sales")
df.agg(F.mean("amount")).show()
```

---

## 📦 Sorting & Sampling
```python
df.orderBy("col")
df.limit(10)
```

---

## 🔗 Joins
```python
df1.join(df2, "key", "left")
df1.union(df2)
```

---

## 🧠 Machine Learning: Feature Engineering
```python
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder

assembler = VectorAssembler(inputCols=["f1", "f2"], outputCol="features")
indexer = StringIndexer(inputCol="label", outputCol="label_idx")
encoder = OneHotEncoder(inputCols=["label_idx"], outputCols=["label_vec"])
```

---

## 🤖 Model Training
```python
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

lr = LogisticRegression(featuresCol="features", labelCol="label_idx")
pipeline = Pipeline(stages=[assembler, indexer, encoder, lr])
model = pipeline.fit(df)
```

---

## 🧪 Model Evaluation
```python
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(labelCol="label_idx", metricName="accuracy")
accuracy = evaluator.evaluate(model.transform(df))
```

---

## 🔁 Cross-Validation
```python
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

paramGrid = ParamGridBuilder().addGrid(lr.maxIter, [10, 20]).build()
cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3)
cvModel = cv.fit(train_data)
```

---

## 🔢 Clustering Example (KMeans)
```python
from pyspark.ml.clustering import KMeans

kmeans = KMeans(k=3, featuresCol="features")
model = kmeans.fit(df)
```

---

## 🔄 Caching & Optimization
```python
df.cache()
df.persist()
df.unpersist()
```

---

## 🗃️ SQL Queries
```python
df.createOrReplaceTempView("table")
spark.sql("SELECT * FROM table WHERE col > 50").show()
```

---

## 🧠 Deep Learning with Spark

### 🔬 TensorFlowOnSpark (Distributed TensorFlow)
```python
from tensorflowonspark import TFCluster
import tensorflow as tf

def map_fun(args, ctx):
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])
    # define model...

cluster = TFCluster.run(sc, map_fun, args, num_executors, num_ps, TensorFlowOnSpark.InputMode.TENSORFLOW)
```

### 🧬 PyTorch + Petastorm
```python
from petastorm.spark import make_spark_converter
from petastorm.pytorch import DataLoader

converter = make_spark_converter(df)
torch_dataloader = DataLoader(converter, batch_size=32)
```

---

## 📝 Natural Language Processing (NLP) with Spark NLP

### 🔡 Basic NLP Pipeline
```python
import sparknlp
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import Tokenizer, Normalizer

document = DocumentAssembler().setInputCol("text").setOutputCol("document")
tokenizer = Tokenizer().setInputCols(["document"]).setOutputCol("token")
```

### 🧠 Named Entity Recognition
```python
from sparknlp.annotator import NerDLModel

ner = NerDLModel.pretrained("ner_dl", "en").setInputCols(["document", "token"]).setOutputCol("ner")
pipeline = Pipeline(stages=[document, tokenizer, ner])
result = pipeline.fit(df).transform(df)
```

---

## 🖼️ Computer Vision with Spark

### 🧪 OpenCV + PySpark UDF
```python
import cv2, numpy as np
from pyspark.sql.functions import udf
from pyspark.sql.types import BinaryType

def process_image(img_bytes):
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    return img_bytes

df = df.withColumn("processed_image", udf(process_image, BinaryType())(df["image"]))
```

### 🔍 Inference using TorchVision
```python
import torch
from torchvision import models, transforms
from PIL import Image
import io

model = models.resnet50(pretrained=True)
model.eval()

def predict_image(img_bytes):
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    tensor = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])(image).unsqueeze(0)
    return torch.argmax(model(tensor)).item()
```

---

## 📚 Additional Resources

### 📘 Official Docs & APIs
- [PySpark Documentation](https://spark.apache.org/docs/latest/api/python/index.html)
- [Apache Spark Docs](https://spark.apache.org/docs/latest/)
- [MLlib (Machine Learning)](https://spark.apache.org/docs/latest/ml-guide.html)
- [Spark SQL Guide](https://spark.apache.org/docs/latest/sql-programming-guide.html)
- [Spark NLP](https://nlp.johnsnowlabs.com/)

### 🎓 Courses & Tutorials
- [Databricks Academy (Free PySpark Training)](https://academy.databricks.com/)
- [Apache Spark on Coursera (by UC Berkeley)](https://www.coursera.org/learn/apache-spark)
- [PySpark Crash Course - YouTube](https://www.youtube.com/watch?v=_C8kWso4ne4)

### 📘 Books
- *Learning Spark: Lightning-Fast Data Analytics* by Jules S. Damji et al.
- *High Performance Spark* by Holden Karau

### 🔧 Tools & Ecosystem
- [TensorFlowOnSpark](https://github.com/yahoo/TensorFlowOnSpark)
- [Petastorm for PyTorch](https://github.com/uber/petastorm)
- [Spark NLP GitHub](https://github.com/JohnSnowLabs/spark-nlp)

---

## ✅ Recommended Best Practices
- Use `persist()` for multiple actions on a DataFrame.
- Avoid shuffling data unnecessarily.
- Partition wisely based on data volume.
- Monitor jobs via Spark UI (`http://<driver-node>:4040`)
- Prefer `DataFrame` APIs over RDDs for performance and optimization.

---

## 📌 License
This cheat sheet is open for educational and project use. Feel free to contribute.

---

## 🤝 Contributions Welcome!
Pull requests, feature additions, and improvements are always welcome! 🎉
