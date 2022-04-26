#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
import pyspark.sql.functions as fn
from pyspark.mllib.evaluation import RankingMetrics


class PopularityBaseline():

    def __init__(self, threshold = None, damping = 0):

        self.threshold = threshold
        self.damping = damping
        self.fitted = False
        self.results = None

    def fit(self, ratings, top_n=100):
        ratings = ratings.groupBy('movieId').agg(fn.sum('rating').alias('tot_rating'), fn.count('userId').alias('counts')) \
            .withColumn('counts', fn.col('counts') + self.damping)
        if self.threshold is not None:
            ratings = ratings.filter(ratings.counts >= self.threshold)
        results = ratings.withColumn('popularity', fn.col('tot_rating') / fn.col('counts')).drop('tot', 'counts').orderBy(fn.desc('popularity'))
        results = results.limit(top_n).select('movieId').rdd.flatMap(lambda x: x).collect()
        self.fitted = True
        self.results = results
        return results

    def evaluate(self, results, test_set):
        data = test_set.filter(test_set.rating > 2.5).drop('rating','timestap')
        UserMovies=[]
        users = [x.userId for x in test_set.select('userId').distinct().collect()]
        for user in users:
            UserMovies.append((results, data.filter('userId'==user).select('movieId').rdd.flatMap(lambda x: x).collect()))
        predictionAndLabels = spark.sparkContext.parallelize(UserMovies)
        metrics = RankingMetrics(predictionAndLabels)
        return metrics

#evaluation done by using Spark RankingMetrics which takes as in put an rdd where each user is a row and for each user have list of recommended movies
#and list of ground truth movies. So for each user get the ground truth as those movies watched with rating > 2.5
