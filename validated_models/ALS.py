# -*- coding: utf-8 -*-

# https://www.timlrx.com/blog/creating-a-custom-cross-validation-function-in-pyspark

import pyspark.sql.functions as fn
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.recommendation import ALS 
from pyspark.sql import functions as fn
from pyspark.ml.evaluation import RankingEvaluator


class CustomCrossValidatorALS():
    
    def __init__(self, seed=0):
        
        self.fitted = False
        self.seed = seed
        self.predictions = None
        self.predsAndlabels = None
        
    def cv_fitted(self, ratings, test_ratings, top_k = 10, rank = [10], regParam = [0.1], maxIter=[10], coldStartStrategy="nan"):
        self.rank = rank
        self.regParam = regParam
        self.maxIter = maxIter
        self.coldStartStrategy = coldStartStrategy
        self.score = 0
        self.predsAndlabels = None
        self.best_predsAndlabels = None
        self.best_score = 0
        self.best_model = None

        als = ALS(userCol="userId",itemCol="movieId",ratingCol="rating", seed = self.seed)

        for reg in regParam:
            for r in rank:
                for i in maxIter:
                    als.setMaxIter(i)
                    als.setRank(r)
                    als.setRegParam(reg)

                    self.model = als.fit(ratings)
                    self.fitted = True

                    self.predictions = self.model.transform(test_ratings).drop('timestamp')
                    self.score, self.predsAndlabels = self.evaluate(self.predictions, top_k)

                    if self.score > self.best_score:
                        self.best_model = self.model
                        self.best_score = self.score
                        self.best_predsAndlabels = self.predsAndlabels

        print('Best ALS model given parameters: ', self.best_model)
        print("Best ALS model score given parameters: ", self.best_score)

        self.score = 0
        self.predsAndlabels = None

        return self.best_model
    
    def evaluate(self, predictions, top_k):
        df_label = predictions.groupBy('userId').agg(fn.collect_list('movieId').alias('label'))

        # Generate top k movie recommendations for each user
        df_recs = self.model.recommendForAllUsers(top_k).withColumn('recommendations', fn.explode((fn.col('recommendations'))))
        df_recs = df_recs.withColumn('recommendations', df_recs.recommendations.getItem('movieId')).groupBy('userId').agg(fn.collect_list('recommendations').alias('recommendations'))
        self.predsAndlabels = df_label.join(df_recs, 'userId').select(fn.col('recommendations').cast('array<double>').alias('recommendations'), fn.col('label').cast('array<double>').alias('label'))

        evaluator = RankingEvaluator().setPredictionCol('recommendations')

        return evaluator.evaluate(self.predsAndlabels), self.predsAndlabels
   