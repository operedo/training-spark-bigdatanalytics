from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("JoinExample")\
        .getOrCreate()

    valuesA = [('A',1),('B',2),('C',3),('D',4)]
    TableA = spark.createDataFrame(valuesA,['name','id'])
 
    valuesB = [('X',1),('A',2),('C',3),('Y',4)]
    TableB = spark.createDataFrame(valuesB,['name','id'])

    ta = TableA.alias('ta')
    tb = TableB.alias('tb')

    # inner join
    inner_join = ta.join(tb, ta.name == tb.name)
    inner_join.show()

    # left join
    left_join = ta.join(tb, ta.name == tb.name,how='left') 
    left_join.show()
    left_join = ta.join(tb, ta.name == tb.name,how='left_outer') 
    left_join.show()

    # right join
    right_join = ta.join(tb, ta.name == tb.name,how='right')
    right_join.show()
    right_join = ta.join(tb, ta.name == tb.name,how='right_outer')
    right_join.show()

    # full outer join
    full_outer_join = ta.join(tb, ta.name == tb.name,how='full')
    full_outer_join.show()
    full_outer_join = ta.join(tb, ta.name == tb.name,how='full_outer')
    full_outer_join.show()

