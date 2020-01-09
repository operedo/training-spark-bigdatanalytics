#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
A simple example demonstrating model selection using CrossValidator.
This example also demonstrates how Pipelines are Estimators.
Run with:

  bin/spark-submit examples/src/main/python/ml/cross_validator.py
"""
from __future__ import print_function

# $example on$
from pyspark.ml import Pipeline,PipelineModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
# $example off$
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("CrossValidatorExample")\
        .getOrCreate()


    bestModel = PipelineModel.load("persistedBestModel.model") 

    # Prepare test documents, which are unlabeled.
    test = spark.createDataFrame([
        (4, "spark i j k"),
        (5, "l m n"),
        (6, "mapreduce spark"),
        (7, "apache hadoop")
    ], ["id", "text"])

    # Make predictions on test documents. cvModel uses the best model found (lrModel).
    prediction = bestModel.transform(test)
    selected = prediction.select("id", "text", "probability", "prediction")
    for row in selected.collect():
        print(row)
    # $example off$

    spark.stop()
