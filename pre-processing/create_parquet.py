def write_parquet_to_hdfs(result):
    try:

        now = datetime.datetime.now()
        date = now.strftime('%Y.%m.%d')


        last_time = result.select("timestamp").orderBy(F.desc("timestamp")).limit(1).rdd.map(
            lambda r: r[0]).collect()[
            0].strftime("%Y-%m-%d %H:%M:%S")
        path = "hdfs://{}:{}".format(config.get('HDFS', 'HOST'), config.get('HDFS', 'PORT'))

        client = InsecureClient(url="http://{}:{}".format(config.get("HDFS", "HOST"),
                                                          config.get("HDFS", "WEB_PORT")),
                                user=config.get("HDFS", "USER"))
        client.write('/sym-log/{}/last_time.txt'.format(date),
                     data=last_time, encoding='utf-8', overwrite=True)

        print('last_time save success')

        checkpoint = get_id_checkpoint(date, config)
        print(checkpoint)

        if checkpoint:
            print("데이터 프레임 length:{}".format(result.count()))
            result_df = result.filter(F.col("timestamp") > checkpoint)
            print("데이터 프레임 length:{}".format(result_df.count()))

            result_df = result_df.coalesce(1).select(F.col('timestamp'), F.col('id'), F.col('hostname'), F.col('path'), F.col('message'), F.col('pid'), F.col('program'),F.col('features'))
            # result_df.write.parquet("{}/sym-log/{}/preprocessing".format(path, date), mode="overwrite")
            result_df.write.parquet("{}/sym-log/{}/preprocessing".format(path, date), mode="append")

    except:
        print('데이터 저장 실패 : ')
        traceback.print_exc()
