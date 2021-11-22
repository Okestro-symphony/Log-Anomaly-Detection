def write_data_to_es():
    try:
        # schema 형식 맞추기
        spark_df = spark_df.withColumn('score',spark_df['score'].cast("float").alias('score'))
        spark_df = spark_df.withColumn('pid', spark_df['pid'].cast("float").alias('pid'))
        spark_df = spark_df.withColumn("is_anomaly", when(F.col("is_anomaly") == -1, 'True').otherwise(F.col('is_anomaly')))
        spark_df = spark_df.withColumn("is_anomaly", when(F.col("is_anomaly") == 1, 'False').otherwise(F.col('is_anomaly')))

        spark_df = spark_df.withColumn("is_anomaly", F.col("is_anomaly").cast('boolean'))

        # host를 위한 처리
        final_df = spark_df.withColumnRenamed('hostname', 'host.hostname') \
            .withColumn('pid', F.when(F.isnan(F.col('pid')), None).otherwise(F.col('pid')))
        
        #final_df.printschema()
        print('shcema change success')

    except Exception as ex:
        print('spark schema 형식 맞춤 실패 : ', ex)

    try:
        # es 데이터 삽입
        now = datetime.datetime.now()
        date = now.strftime('%Y.%m.%d')
        spark.save_to_es(final_df, 'sym-anomaly-log-{}'.format(date))
        print('data save')
    except Exception as ex:
        print('ES 데이터 적재 실패 : ', ex)

if __name__ == '__main__':
    write_data_to_es()
