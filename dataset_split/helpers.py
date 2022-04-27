
# -*- coding: utf-8 -*-
import warnings
from pyspark.sql.window import Window
from pyspark.sql import functions as fn
from random import sample


def readRDD(spark, dirstring, small, column_name):
    #TODO instead of manually writing this in, column_names should be read in from spark methods? A lot of repetition here
    if small == True:
        print('Using small set...')
        print('')
        dir = dirstring + '/ml-latest-small/'
        column_names = ['movies','ratings','links','tags']
        # Load into DataFrame
        if column_name in column_names:
            if column_name == 'movies':
               return spark.read.csv(dir + 'movies.csv', header=True, schema='movieId INT, title STRING, genres STRING'), column_name
            elif column_name == 'links':
                return spark.read.csv(dir + 'links.csv', header=True, schema='movieId INT, imdbId FLOAT, tmdbId FLOAT'), column_name
            elif column_name == 'ratings':
                return spark.read.csv(dir + 'ratings.csv', header=True, schema='userId INT, movieId INT, rating FLOAT, timestamp INT'), column_name
            elif column_name == 'tags':
                return spark.read.csv(dir + 'tags.csv', header=True, schema='userId INT, movieId INT, tag STRING, timestamp INT'), column_name
        else:
            warnings.warn('Warning Message: Column name not found; opting for ratings')
            return spark.read.csv(dir + 'ratings.csv', header=True, schema='userId INT, movieId INT, rating FLOAT, timestamp INT'), 'ratings'
    else:
        print('Using large set...')
        print('')
        dir = dirstring + '/ml-latest/'
        column_names = ['movies','ratings','links','tags','genome-tags','genome-scores']
        # Load into DataFrame
        if column_name in column_names:
            if column_name == 'movies':
               return spark.read.csv(dir + 'movies.csv', header=True, schema='movieId INT, title STRING, genres STRING'), column_name
            elif column_name == 'links':
                return spark.read.csv(dir + 'links.csv', header=True, schema='movieId INT, imdbId FLOAT, tmdbId FLOAT'), column_name
            elif column_name == 'ratings':
                return spark.read.csv(dir + 'ratings.csv', header=True, schema='userId INT, movieId INT, rating FLOAT, timestamp INT'), column_name
            elif column_name == 'tags':
                return spark.read.csv(dir + 'tags.csv', header=True, schema='userId INT, movieId INT, tag STRING, timestamp INT'), column_name
            elif column_name == 'genome-scores':
                return spark.read.csv(dir + 'genome-scores.csv', header=True, schema='movieId INT, tagId INT, relevance STRING'), column_name
            elif column_name == 'genome-tags':
                return spark.read.csv(dir + 'genome-tags.csv', header=True, schema='tagId INT, tag STRING'), column_name
        else:
            warnings.warn('Warning Message: Column name not found; opting for ratings')
            return spark.read.csv(dir + 'ratings.csv', header=True, schema='userId INT, movieId INT, rating FLOAT, timestamp INT'), 'ratings'

def ratings_split(rdd, prop):
    windowSpec  = Window.partitionBy('userId').orderBy('timestamp')

    ratings = rdd \
            .withColumn('row_number', fn.row_number().over(windowSpec)) \
            .withColumn('n_ratings', fn.count('rating').over(windowSpec)) \
            .withColumn('prop_idx', (fn.col('row_number') / fn.col('n_ratings')))
    ratings.show(20)

    ratings_train = ratings.filter(ratings.prop_idx <= prop).drop('row_number', 'n_ratings', 'prop_idx')

    ratings_val_test = ratings.filter(ratings.prop_idx > prop).drop('row_number', 'n_ratings', 'prop_idx')
    distinct_user_ids = [x.userId for x in ratings_val_test.select('userId').distinct().collect()]

    ratings_validation = ratings_val_test.filter(ratings.userId.isin(sample(distinct_user_ids, len(distinct_user_ids)//2)))
    ratings_test = ratings_val_test.subtract(ratings_validation)

    return ratings_train, ratings_test, ratings_validation