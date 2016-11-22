package TelCom

import com.sun.glass.ui.Application
import org.apache.spark.sql.Row
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.{SparkContext, SparkConf}

/**
 * Hello world!
 *
 */
object ETL
{
  //获得t之前i被所有用户浏览的次数
  //t之前i被所有用户查看的次数
  //t之前i被所有用户点赞的次数
  //t之前i被所有用户踩的次数
  def main(args: Array[String]): Unit = {

    //数据源
    val conf = new SparkConf()
    conf.setAppName("ETL")
    val sc = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    val sqlContext2: HiveContext = new HiveContext(sc)
    val adddataDataFrame = sqlContext2.sql("select * from addata")

    //获得当前需要统计的用户和帖子
    var data = sc.textFile("E:\\电信学院_数据挖掘赛数据_contest_data\\race_data")
//    data = data.map(x => x.split("/t")).map(x => Row(x(0),"b",x(2),x(3))).map(x =>
//    {
//
//
//    })



    val hiveContext = new org.apache.spark.sql.hive.HiveContext(sc)

    var dev_praise = hiveContext.sql("SELECT dev_id,COUNT(post_id) as count1 FROM second_train_data  WHERE praise = \"1\" GROUP BY dev_id having count1 > 10 ORDER BY count1 DESC;")
    var post_data = hiveContext.sql("SELECT * from second_post_data;")
    var dev_praise_content = post_data.join(dev_praise,post_data("dev_id") === dev_praise("dev_id"))


    //    val result = data.map(x = )

  }


}
