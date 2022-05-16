# -*- coding: utf-8 -*-

# https://www.timlrx.com/blog/creating-a-custom-cross-validation-function-in-pyspark

import pyspark.sql.functions as fn
from pyspark.ml.recommendation import ALS
from pyspark.sql import functions as fn
from pyspark.ml.evaluation import RankingEvaluator


class ValidatedALS():

    def __init__(self, seed=0):

        self.fitted = False
        self.seed = seed
        self.predictions = None
        self.predsAndlabels = None
        self.score = 0

    def validate(self, ratings_train, ratings_val, top_k=100, rank=[10], regParam=[0.1], maxIter=[10], coldStartStrategy="nan", verbose=True, metric='meanAveragePrecision'):
        
        self.k = top_k
        self.coldStartStrategy = coldStartStrategy
        
        best_predsAndlabels = None
        best_score = 0
        best_model = None
        best_rank = None
        best_maxIter = None
        best_regParam = None

        als = ALS(userCol="userId", itemCol="movieId",
                  ratingCol="rating", seed=self.seed)

        for reg in regParam:
            for r in rank:
                for i in maxIter:
                    als.setMaxIter(i)
                    als.setRank(r)
                    als.setRegParam(reg)

                    if verbose:
                        print("Fitting ALS model given parameters: Rank {r} | MaxIter: {i} | RegParam: {reg} | ColdStartStrategy: {strat} |".format(
                           r=r, i=i, reg=reg, strat=self.coldStartStrategy))

                    self.model = als.fit(ratings_train)
                    self.fitted = True

#                    self.predictions = self.model.transform(
#                        self.test_ratings).drop('timestamp')
#                    self.score, self.predsAndlabels = self.evaluate(
#                        self.predictions, top_k, self.metric)
                    self.score, self.predsAndlabels = self.evaluate(ratings_val, top_k, metric)

                    if verbose:
                        if metric == 'meanAveragePrecision':
                            print('Score for ALS model given these parameters using MAP@{k}: {s}'.format(k = top_k, s = self.score))
                            print('------------------------------------------------------------------------------------------------------------------------------')     
                        elif metric == 'ndcgAtK':
                            print('Score for ALS model given these parameters using NCDG@{k}: {s}'.format(k = top_k, s = self.score))
                            print('------------------------------------------------------------------------------------------------------------------------------')     


                    if self.score > self.best_score:
                        best_model = self.model
                        best_score = self.score
                        best_predsAndlabels = self.predsAndlabels
                        best_rank = r
                        best_maxIter = i
                        best_regParam = reg

        print('==============================================================================================================================')
        if metric == 'meanAveragePrecision':
            print("Best ALS model given parameters using MAP@{k}: Rank {r} | MaxIter: {i} | RegParam: {reg} | ColdStartStrategy: {strat} |".format(
                k=top_k, r= best_rank, i=best_maxIter, reg=best_regParam, strat=self.coldStartStrategy))
            print("Best ALS model score given these parameters: ", best_score)
        else: 
            print("Best ALS model given parameters using NCDG@{k}: Rank {r} | MaxIter: {i} | RegParam: {reg} | ColdStartStrategy: {strat} |".format(
                k=top_k, r= best_rank, i=best_maxIter, reg=best_regParam, strat=self.coldStartStrategy))
            print("Best ALS model score given these parameters: ", best_score)

        self.model = best_model
        self.score = best_score
        self.predsAndlabels = best_predsAndlabels
        self.rank = best_rank
        self.maxIter = best_maxIter
        self.regParam = best_regParam
        
        return self.model

    def evaluate(self, ratings_test, top_k, metricName='meanAveragePrecision'):
#        df_label = predictions.groupBy('userId').agg(
#            fn.collect_list('movieId').alias('label'))
        
        data = ratings_test.filter(ratings_test.rating > 2.5).drop(
            'rating', 'timestamp')
        UserMovies = data.groupBy('userId').agg(
            fn.collect_list('movieId').alias('label'))

        # Generate top k movie recommendations for each user
        df_recs = self.model.recommendForAllUsers(top_k).withColumn(
            'recommendations', fn.explode((fn.col('recommendations'))))
        df_recs = df_recs.withColumn('recommendations', df_recs.recommendations.getItem(
            'movieId')).groupBy('userId').agg(fn.collect_list('recommendations').alias('recommendations'))
        
#        self.predsAndlabels = df_label.join(df_recs, 'userId').select(fn.col('recommendations').cast(
#            'array<double>').alias('recommendations'), fn.col('label').cast('array<double>').alias('label'))
        
        self.predsAndlabels = df_recs.join(UserMovies, 'userId').select(fn.col('recommendations').cast(
            'array<double>').alias('recommendations'), fn.col('label').cast('array<double>').alias('label'))
        

        if metricName == 'meanAveragePrecision':
            evaluator = RankingEvaluator(metricName=metricName).setPredictionCol('recommendations')
        elif metricName == 'ndcgAtK':
            evaluator = RankingEvaluator(metricName=metricName).setPredictionCol('recommendations').setK(top_k)

        return evaluator.evaluate(self.predsAndlabels), self.predsAndlabels
