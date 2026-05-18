import org.apache.spark.sql.{SparkSession, Row}
import org.apache.spark.SparkConf
import org.graphframes.GraphFrame
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._

object task1 {
  def main(args: Array[String]): Unit = {
    val filterThreshold = args(0).toInt
    val inputFilePath = args(1)
    val outputFilePath = args(2)

    // Configure Spark
    val conf = new SparkConf().setAppName("task1")
    val spark = SparkSession.builder().config(conf).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    val startTime = System.nanoTime()

    // Load data
    val df = spark.read.option("header", "true").csv(inputFilePath)
    val userBusinessRDD: RDD[(String, String)] = df.select("user_id", "business_id")
      .rdd.map(row => (row.getString(0), row.getString(1)))

    // Group businesses by user
    val userBusinessSetRDD = userBusinessRDD.groupByKey().mapValues(_.toSet)

    // Generate user pairs and filter based on the threshold
    val userPairsRDD = userBusinessSetRDD.cartesian(userBusinessSetRDD)
    val edgesRDD = userPairsRDD.filter { case ((user1, businesses1), (user2, businesses2)) =>
      user1 != user2 && businesses1.intersect(businesses2).size >= filterThreshold
    }.map { case ((user1, _), (user2, _)) => (user1, user2) }.distinct()

    // Convert edges to DataFrame
    import spark.implicits._
    val edgesDF = edgesRDD.toDF("src", "dst")

    // Create nodes DataFrame from edges
    val nodesDF = edgesDF.select("src").union(edgesDF.select("dst")).distinct().withColumnRenamed("src", "id")

    // Create GraphFrame
    val graph = GraphFrame(nodesDF, edgesDF)

    // Detect communities using Label Propagation Algorithm (LPA)
    val communities = graph.labelPropagation.maxIter(5).run()

    // Process communities and write to output file
    val sortedCommunities = communities.select("label", "id").rdd
      .map(row => (row.getLong(0), row.getString(1)))
      .groupByKey()
      .mapValues(users => users.toList.sorted)
      .sortBy { case (_, users) => (users.size, users) }
      .map { case (_, users) => users.map("'" + _ + "'").mkString(", ") }
      .collect()

    val file = new java.io.PrintWriter(outputFilePath)
    try {
      sortedCommunities.foreach(line => file.println(line))
    } finally {
      file.close()
    }

    val duration = (System.nanoTime() - startTime) / 1e9d
    println(f"Duration: $duration%.2f seconds")

    spark.stop()
  }
}
