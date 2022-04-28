#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pyspark.sql.functions as fn
from pyspark.ml.recommendation import ALS 
from pyspark.ml.evaluation import RankingEvaluator


class ALS():
    
    def __init__(self, rank = 5, regParam = 0.1, maxIter=10, coldStartStrategy="nan", seed=0):
        
        self.rank = rank
        self.regParam = regParam
        self.fitted = False
        self.results = None
        self.maxIter = maxIter
        self.coldStartStrategy = coldStartStrategy
        self.seed = seed
        self.model = ALS(userCol="userId", itemCol="movieId", ratingCol="rating", rank=rank, regParam=regParam, maxIter=maxIter, coldStartStrategy=coldStartStrategy, seed=seed)
        
    def fit(self, ratings):
        self.model = self.model.fit(ratings)
        predictions = self.model.transform(ratings).drop('timestamp')
        #self.predictions.show()

        self.fitted = True
        self.results = predictions
        return predictions
    
    def evaluate(self, results, test_set, n_items=10):
        df_label = self.results.groupBy('userId').agg(fn.collect_list('movieId').alias('label'))

        # Generate top n_items movie recommendations for each user
        df_recs = self.model.recommendForAllUsers(n_items).withColumn('recommendations', fn.explode((fn.col('recommendations'))))
        df_recs = df_recs.withColumn('recommendations', df_recs.recommendations.getItem('movieId'))\
                    .groupBy('userId').agg(fn.collect_list('recommendations').alias('recommendations'))

        predsAndlabels = df_label.join(df_recs, 'userId').select('recommendations', 'label')

        predsAndlabels.show()

        evaluator = RankingEvaluator()
        evaluator.setPredictionCol("recommendations")

        return evaluator

#evaluation done by using Spark RankingMetrics which takes as in put an rdd where each user is a row and for each user have list of recommended movies
#and list of ground truth movies. So for each user get the ground truth as those movies watched with rating > 2.5
