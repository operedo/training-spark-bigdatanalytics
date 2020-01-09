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

from __future__ import print_function

# $example on$
from pyspark.ml.feature import SQLTransformer
# $example off$
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("SQLTransformerExample")\
        .getOrCreate()

    # $example on$
    df = spark.createDataFrame([
        (0, 1.0, 3.0),
        (1, 1.0, 3.0),
        (2, 1.0, 0.0),
        (3, 1.0, 1.0),
        (4, 0.0, 1.0),
        (5, 0.0, 1.0),
        (6, 0.0, 2.0),
        (7, 0.0, 2.0),
        (8, 2.0, 2.0),
        (9, 2.0, 3.0),
        (10, 3.0, 3.0),
        (11, 3.0, 3.0),
        (12, 3.0, 4.0),
        (13, 4.0, 3.0),
        (14, 4.0, 2.0),
        (15, 4.0, 1.0),
        (16, 5.0, 0.0),
        (17, 5.0, 0.0),
        (18, 5.0, 0.0),
        (19, 4.0, 0.0),
        (20, 4.0, 1.0),
        (21, 2.0, 2.0),
        (22, 2.0, 3.0),
        (23, 2.0, 5.0)
    ], ["id", "v1", "v2"])
    df.show()
    sqlTrans = SQLTransformer(
        statement="SELECT *, (v1 + v2) AS v3, (v1 * v2) AS v4, AVG(v1) OVER (ORDER BY id ASC ROWS 9 PRECEDING) AS MAv1,  AVG(v2) OVER (ORDER BY id ASC ROWS 9 PRECEDING) AS MAv2 FROM __THIS__")
    sqlTrans.transform(df).show()
    # $example off$

    spark.stop()
