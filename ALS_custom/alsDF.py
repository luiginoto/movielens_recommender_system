# -*- coding: utf-8 -*-
from . import helpers
import numpy as np

from pyspark.ml.recommendation import ALS

def validated_ALS(self, spark, train, test, val, rank=10, maxIter=10, regParam=0.1, userCol='userId', itemCol='movieId', ratingCol='rating', coldStartStrategy='drop')
	
    als = ALS(rank=rank, maxIter=maxIter, regParam=regParam, userCol=userCol, itemCol=itemCol, ratingCol=ratingCol, coldStartStrategy=coldStartStrategy)
    self.model = als.fit(spark.createDataFrame(train))

    global_user = None
    user_list = np.array([], dtype=np.float64)
    testItems = list()
    for row in test.iterrows():

        if row[1]['user'] != global_user:
            user_list = np.append(user_list, row[1]['user'])
            testItems.append(int(row[1]['item']))
            global_user = row[1]['user']
        else:
            testItems.append(int(row[1]['item']))
        true_list[global_user] = testItems

    pandasDf = pd.DataFrame({'user': user_list})
    sub_user = spark.createDataFrame(pandasDf)
    labelsList = list()
    for user, items in self.model.recommendForUserSubset(sub_user, 30).collect():

        predict_items = [i.item for i in items]
        labelsList.append((predict_items, true_list[user]))
    labels = spark.sparkContext.parallelize(labelsList)
    metrics = RankingMetrics(labels)
    print(metrics.meanAveragePrecision)
   