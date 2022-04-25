#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Script used to split the MovieLens dataset into training, validation and
test sets
'''

import sys
# import os

from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql import functions as fn

def main(spark, in_path, out_path):
    
    # path_basename = os.path.basename(os.path.normpath(in_path))
    
    ratings = spark.read.csv(in_path,
                             header=True,
                             schema='userId INT, movieId INT, rating FLOAT, timestamp INT')
    
    print('Printing ratings schema')
    ratings.printSchema()
    
    windowSpec  = Window.partitionBy('userId').orderBy('timestamp')
    
    ratings = ratings \
            .withColumn('row_number', fn.row_number().over(windowSpec)) \
            .withColumn('n_ratings', fn.count('rating').over(windowSpec)) \
            .withColumn('prop_idx', (fn.col('row_number') / fn.col('n_ratings')))
    ratings.show()
    
    ratings_train = ratings.filter(ratings.prop_idx <= 0.8)
    ratings_validation = ratings.filter((ratings.prop_idx > 0.8) & (ratings.prop_idx <= 0.9)) 
    ratings_test = ratings.filter((ratings.prop_idx > 0.9) & (ratings.prop_idx <= 1)) 
    
    ratings_train.write.csv(f'{out_path}/ratings_train.csv')
    ratings_validation.write.csv(f'{out_path}/ratings_validation.csv')
    ratings_test.write.csv(f'{out_path}/ratings_test.csv')



# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('part1').getOrCreate()

    # Get path of ratings file
    in_path = sys.argv[1]
    
    # Get destination directory of training, validation and test set files
    out_path = sys.argv[2]

    # Call our main routine
    main(spark, in_path, out_path)

