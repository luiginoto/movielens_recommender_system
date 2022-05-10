# -*- coding: utf-8 -*-

# https://www.timlrx.com/blog/creating-a-custom-cross-validation-function-in-pyspark

import pyspark.sql.functions as fn
from pyspark.ml.recommendation import ALS
from pyspark.mllib.evaluation import RankingMetrics


class CustomALS():

    def __init__(self, rank=10, regParam=0.1, maxIter=10, seed=0):

        self.model = ALS(rank=rank, regParam=regParam, maxIter=maxIter, userCol="userId", itemCol="movieId",
                  ratingCol="rating", seed=seed)
        self.rank = rank
        self.regParam = regParam
        self.maxIter = maxIter
        self.seed = seed
        self.fitted = False
        self.fitted_model = None
        self.preds = None
        self.predsAndlabels = None

    def fit(self, ratings, top_n=100):
        self.fitted_model = self.model.fit(ratings)
        self.fitted = True
        df_recs = self.fitted_model.recommendForAllUsers(top_n).withColumn(
            'recommendations', fn.explode((fn.col('recommendations'))))
        self.preds = df_recs.withColumn('recommendations', df_recs.recommendations.getItem(
                'movieId')).groupBy('userId').agg(fn.collect_list('recommendations').alias('recommendations'))

    def evaluate(self, test_set):
        data = test_set.filter(test_set.rating > 2.5).drop(
            'rating', 'timestamp')
        UserMovies = data.groupBy('userId').agg(
            fn.collect_list('movieId').alias('liked_movies'))

        self.predsAndlabels = self.preds.join(UserMovies, 'userId').select('recommendations', 'liked_movies').rdd.map(tuple)

        metrics = RankingMetrics(self.predsAndlabels)

        return metrics
