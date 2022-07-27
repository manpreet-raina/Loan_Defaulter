import findspark

findspark.init()
import pyspark
from pyspark import SparkContext
from pyspark.sql import SQLContext


class SparkFunctions:

    def __init__(self):
        self.sc, self.sqlc = self.spark_context()

    def spark_context(self):
        conf = pyspark.SparkConf().set("spark.jars.packages",
                                       "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1").setMaster(
            "local").setAppName("Data Connection").setAll(
            [("spark.driver.memory", "40g"), ("spark.executor.memory", "50g")])
        sc = SparkContext(conf=conf)
        sqlc = SQLContext(sc)
        return sc, sqlc
