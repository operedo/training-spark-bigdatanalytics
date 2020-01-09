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
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
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

    # $example on$
    # Prepare training documents, which are labeled.
    training = spark.createDataFrame([
        ("a b c d e spark", 1.0),
        ("b d", 0.0),
        ("spark f g h", 1.0),
        ("hadoop mapreduce", 0.0),
        ("b spark who", 1.0),
        ("g d a y", 0.0),
        ("spark fly", 1.0),
        ("was mapreduce", 0.0),
        ("e spark program", 1.0),
        ("a e c l", 0.0),
        ("spark compile", 1.0),
        ("hadoop software", 0.0)
    ], ["words", "label"])

    # Configure an ML pipeline, which consists of tree stages: tokenizer, hashingTF, and lr.
    tokenizer = Tokenizer(inputCol="words", outputCol="wordsToken")
    hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
    lr = LogisticRegression(maxIter=10)
    pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])

    # We now treat the Pipeline as an Estimator, wrapping it in a CrossValidator instance.
    # This will allow us to jointly choose parameters for all Pipeline stages.
    # A CrossValidator requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
    # We use a ParamGridBuilder to construct a grid of parameters to search over.
    # With 3 values for hashingTF.numFeatures and 2 values for lr.regParam,
    # this grid will have 3 x 2 = 6 parameter settings for CrossValidator to choose from.
    paramGrid = ParamGridBuilder() \
        .addGrid(hashingTF.numFeatures, [10, 100, 1000]) \
        .addGrid(lr.regParam, [0.1, 0.01]) \
        .build()

    crossval = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=paramGrid,
                              evaluator=BinaryClassificationEvaluator(),
                              numFolds=2)  # use 3+ folds in practice

    # Run cross-validation, and choose the best set of parameters.
    cvModel = crossval.fit(training)

    # Prepare test documents, which are unlabeled.
    test = spark.createDataFrame([
        ("spark i j k"),
        ("l m n"),
        ("mapreduce spark"),
        ("apache hadoop")
    ])

    # Make predictions on test documents. cvModel uses the best model found (lrModel).
    prediction = cvModel.transform(test)
    #selected = prediction.select("words", "probability", "prediction")
    selected = prediction.select("_c0", "probability", "prediction")
    for row in selected.collect():
        print(row)
    # $example off$

    bestModel = cvModel.bestModel
    bestModel.write().overwrite().save("persistedBestModel.model")


    spark.stop()
