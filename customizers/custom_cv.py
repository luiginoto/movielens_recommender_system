# -*- coding: utf-8 -*-

# https://www.timlrx.com/blog/creating-a-custom-cross-validation-function-in-pyspark
import numpy as np
from multiprocessing.pool import ThreadPool
from pyspark.sql.functions import col
import numpy as np
from pyspark.ml import Estimator
from pyspark import keyword_only
from pyspark.ml import Estimator, 
from pyspark.ml.param import Params, Param, TypeConverters
from pyspark.ml.param.shared import HasParallelism
from pyspark.ml.util import (
    MLReadable,
    MLWritable
)
from pyspark.sql.functions import col

class CustomCrossValidator(Estimator, ValidatorParams, HasParallelism, MLReadable, MLWritable):
    """
    Modifies CrossValidator allowing custom train and test dataset to be passed into the function
    Bypass generation of train/test via numFolds
    instead train and test set is user-defined
    """

    splitWord = Param(Params._dummy(), "splitWord", "Tuple to split train and test set e.g. ('train', 'test')",
                      typeConverter=TypeConverters.toListString)
    cvCol = Param(Params._dummy(), "cvCol", "Column name to filter train and test list",
                      typeConverter=TypeConverters.toString)

    @keyword_only
    def __init__(self, estimator=None, estimatorParamMaps=None, evaluator=None,
                 splitWord = ('train', 'test'), cvCol = 'cv', seed=None, parallelism=1):

        super(CustomCrossValidator, self).__init__()
        self._setDefault(parallelism=1)
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def _fit(self, dataset):
        est = self.getOrDefault(self.estimator)
        epm = self.getOrDefault(self.estimatorParamMaps)
        numModels = len(epm)
        eva = self.getOrDefault(self.evaluator)
        nFolds = len(dataset)
        seed = self.getOrDefault(self.seed)
        metrics = [0.0] * numModels
        matrix_metrics = [[0 for x in range(nFolds)] for y in range(len(epm))]

        pool = ThreadPool(processes=min(self.getParallelism(), numModels))

        for i in range(nFolds):
            validation = dataset[list(dataset.keys())[i]].filter(col(self.getOrDefault(self.cvCol))==(self.getOrDefault(self.splitWord))[0]).cache()
            train = dataset[list(dataset.keys())[i]].filter(col(self.getOrDefault(self.cvCol))==(self.getOrDefault(self.splitWord))[1]).cache()

            print('fold {}'.format(i))
            tasks = _parallelFitTasks(est, train, eva, validation, epm)
            for j, metric in pool.imap_unordered(lambda f: f(), tasks):
                # print(j, metric)
                matrix_metrics[j][i] = metric
                metrics[j] += (metric / nFolds)
            # print(metrics)
            validation.unpersist()
            train.unpersist()

        if eva.isLargerBetter():
            bestIndex = np.argmax(metrics)
        else:
            bestIndex = np.argmin(metrics)

        for i in range(len(metrics)):
            print(epm[i], 'Detailed Score {}'.format(matrix_metrics[i]), 'Avg Score {}'.format(metrics[i]))

        print('Best Model: ', epm[bestIndex], 'Detailed Score {}'.format(matrix_metrics[bestIndex]),
              'Avg Score {}'.format(metrics[bestIndex]))

        ### Do not bother to train on full dataset, just the latest train supplied
        # bestModel = est.fit(dataset, epm[bestIndex])
        bestModel = est.fit(train, epm[bestIndex])
        return self._copyValues(CrossValidatorModel(bestModel, metrics))



    '''
    def __init__(self, rank = [5], regParam = [0.1], maxIter=[10], coldStartStrategy="nan", seed=0):
        
        self.rank = rank
        self.regParam = regParam
        self.fitted = False
        self.results = None
        self.maxIter = maxIter
        self.coldStartStrategy = coldStartStrategy
        self.seed = seed
        self.model = None
        
    def cv_fit(self, ratings):

        ParamGridBuilder() \
        .addGrid(lr.regParam, [1.0, 2.0]) \
        .addGrid(lr.maxIter, [1, 5]) \
        .build()

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

        predsAndlabels = df_label.join(df_recs, 'userId').select(fn.col('recommendations').cast('array<double>').alias('recommendations'), fn.col('label').cast('array<double>').alias('label'))

        predsAndlabels.show()

        evaluator = RankingEvaluator()
        evaluator.setPredictionCol("recommendations")

        return evaluator
    '''
    

#evaluation done by using Spark RankingMetrics which takes as in put an rdd where each user is a row and for each user have list of recommended movies
#and list of ground truth movies. So for each user get the ground truth as those movies watched with rating > 2.5
