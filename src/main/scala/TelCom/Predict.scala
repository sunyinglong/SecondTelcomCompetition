package TelCom


import org.apache.spark.mllib.linalg.{DenseVector, Vectors, Vector}
import org.apache.spark.sql.catalyst.expressions.Row
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
import org.apache.spark.mllib.linalg.{DenseVector, Vectors, Vector}
/**
 * Created by Sun on 2016/6/23.
 */
object Predict {
  val sizeOfSample:Int = 22
  def main(args: Array[String]): Unit = {


    val conf = new SparkConf()
    conf.setAppName("logic")
    val sc = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)

    var fileRoot = "H:\\电信学院_数据挖掘赛数据_contest_data"
    val data = sc.textFile(fileRoot + "/toPredict.txt")
    val Model1 = LogisticRegressionModel.load(sc, fileRoot + "/model")
    val Model2 = LogisticRegressionModel.load(sc, fileRoot + "/model")

    //过滤点没有的
    val feature1 = data.map(x => x.split("\t")).map(x =>toPredict(x,Model1,Model2)).filter(x => x != "Normal")
    print(feature1)
    feature1.randomSplit(Array(1, 0), seed = 11L)(0).repartition(1).saveAsTextFile(fileRoot + "/out")

  }

  def toPredict(x:Array[String],model1:LogisticRegressionModel,model2: LogisticRegressionModel): String =
  {
    var z:Array[Double] = new Array[Double](sizeOfSample  - 3)
    for(i <-  2 to x.size-1)
      z(i-2) = x(i).toDouble

    val v:Vector = Vectors.dense(z)
    val result1 =  model1.predict(v)
    val result2 =  model1.predict(v)
    var returnResult:String = null
    if (result1 > 0.95){
      returnResult = x(0).toString + "\t" + x(1).toString + "\t" + "1"}
  else if (result2 > 1){
    returnResult = x(0).toString + "\t" + x(1).toString + "\t" + "2"}
  else
  {returnResult = "Normal"
  }

  return returnResult

  }

}
