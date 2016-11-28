package TelCom


import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
import org.apache.spark.mllib.regression.LabeledPoint

import org.apache.spark.mllib.linalg.{DenseVector, Vectors, Vector}
/**
 * Created by Sun on 2016/6/23.
 */
object LogisticRegressionSupport {

  val sizeOfSample:Int = 22

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf()
    conf.setAppName("logic")
    val sc = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)

    var fileRoot = "H:\\电信学院_数据挖掘赛数据_contest_data"

    // Load training data in LIBSVM format.
    val data = sc.textFile(fileRoot + "\\sample_v1_d.txt")
    val test = sc.textFile(fileRoot + "\\test2.txt")


    // Split data into training (60%) and test (40%).
    //dev_id              	string
//    post_id             	string
//      praise              	string
//    type                	string
//    post_stat_view      	double
//      post_stat_click     	double
//      view_sum            	double
//      click_sum           	double
//      post_view_sum       	bigint
//      ok_num              	bigint
//      no_num              	bigint
//      topic1              	string
//      topic2              	string
//      topic3              	string
//      topic4              	string
//      topic5              	string
//      topic6              	string
//      topic7              	string
//      topic8              	string
//      topic9              	string
//      topic10             	string

    val training = data.map(x => x.split("\t")).filter(x => x.size == 22).
      map(x => {
        var z:Array[Double] = new Array[Double](sizeOfSample - 2)
        for(i <-  2 to sizeOfSample-1)
          if (x(i) != null && x(i) != "null" && x(i) != "")
            z(i-2) = x(i).toDouble
          else
            z(i-2) = 0.0
        z

      })
        .map(x => LabeledPoint(getLebel(x(0)), Vectors.dense(getSubString(x))))


    // Run training algorithm to build the model
    val model1 = new LogisticRegressionWithLBFGS().setNumClasses(2).run(training)
    val model2 = new LogisticRegressionWithLBFGS().setNumClasses(10).run(training)

    //输出阈值
    model1.clearThreshold()

    // 预测模型1 预测模型2.
    val predictionAndLabels1 = training.map { case LabeledPoint(label, features) =>
      val prediction = model1.predict(features)
      (prediction, label)
    }

    //预测负1
//    val predictionAndLabels2 = test.map { case LabeledPoint(label, features) =>
//      val prediction = model1.predict(features)
//      (prediction, label)
//    }

    // Save and load model
    model1.save(sc, fileRoot + "\\model")
//    val sameModel = LogisticRegressionModel.load(sc, "myModelPath")
    print(model1.weights)

  }

  def getSubString(x:Array[Double]) : Array[Double]={

    var z:Array[Double] = new Array[Double](sizeOfSample - 3)
    for(i <-  1 to sizeOfSample-3)
      z(i-1) = x(i)
    return z

  }


  def getLebel(x:Double) : Double ={
    if(x != 1&& x != 0)
        0.0
    else
      x
  }



  //j进行结果结算
  //val metrics = new MulticlassMetrics(predictionAndLabels1)
  //val precision = metrics.precision




}