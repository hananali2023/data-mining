import org.apache.spark.{SparkConf, SparkContext}
import scala.util.Random
import scala.collection.mutable
import java.io._



val conf = new SparkConf().setAppName("task1")
val sc = new SparkContext(conf)



val inputFile = args(0)
val outputFile = args(1)



val data = sc.textFile(inputFile).filter(row => row != "user_id,business_id,stars").map(line => line.split(","))



val allUsers = data.map(_(0))
val allBusinesses = data.map(_(1))

val businessUsersRDD = data.map(x => (x(1), x(0))).groupByKey().mapValues(_.toSet)
val distinctUsers = data.map(_(0)).distinct().collect()
val userIndex = distinctUsers.zipWithIndex.toMap



def hashFunctions(numFuncs: Int, maxVal: Int): Seq[(Int, Int)] = {
  (0 until numFuncs).map(_ => (Random.nextInt(maxVal), Random.nextInt(maxVal)))
}



def createSignatureMatrix(
  businessUsers: Iterable[(String, Set[String])],
  numFuncs: Int,
  userIndex: Map[String, Int],
  bins: Int,
  prime: Int
): Map[String, Seq[Int]] = {
  val hashFuncs = hashFunctions(numFuncs, bins)
  businessUsers.map { case (business, users) =>
    val signature = hashFuncs.map { case (a, b) =>
      users.map(user => ((a * userIndex(user) + b) % prime) % bins).min
    }
    business -> signature
  }.toMap
}



def lsh(signatures: Map[String, Seq[Int]], bandSize: Int): Map[Seq[Int], Seq[String]] = {
  val lshBuckets = mutable.Map[Seq[Int], mutable.Buffer[String]]().withDefaultValue(mutable.Buffer())
  for ((business, signature) <- signatures) {
    for (i <- 0 until signature.length by bandSize) {
      val band = signature.slice(i, i + bandSize)
      lshBuckets(band) += business
    }
  }
  lshBuckets.map { case (k, v) => (k, v.toSeq) }.toMap
}



def candidatePairs(lshBuckets: Map[Seq[Int], Seq[String]]): Set[(String, String)] = {
  lshBuckets.values.flatMap { bucket =>
    if (bucket.length > 1) bucket.combinations(2).map { case Seq(a, b) => (a, b) }
    else Seq.empty
  }.toSet
}




def jaccardSimilarity(
  candidatePairs: Set[(String, String)],
  businessUsers: Map[String, Set[String]],
  threshold: Double
): Map[(String, String), Double] = {
  candidatePairs.collect {
    case (bus1, bus2) if bus1 != bus2 =>
      val users1 = businessUsers(bus1)
      val users2 = businessUsers(bus2)
      val intersection = users1.intersect(users2).size
      val union = users1.union(users2).size
      val jaccardSim = intersection.toDouble / union
      if (jaccardSim >= threshold) (bus1, bus2) -> jaccardSim
  }.toMap
}



val businessUsersDict = businessUsersRDD.collectAsMap().toMap
val numHashes = 60
val prime = 987654321
val hashFuncs = hashFunctions(numHashes, distinctUsers.length)
val signatureMatrix = createSignatureMatrix(businessUsersDict, numHashes, userIndex, distinctUsers.length, prime)
val bandSize = 2
val buckets = lsh(signatureMatrix, bandSize)
val pairs = candidatePairs(buckets)
val similarityThreshold = 0.5
val similarPairs = jaccardSimilarity(pairs, businessUsersDict, similarityThreshold)



val writer = new PrintWriter(new File(outputFile))
writer.write("business_id_1,business_id_2,similarity\n")
similarPairs.toSeq.sortBy(_._1).foreach { case ((bus1, bus2), sim) =>
  writer.write(s"$bus1,$bus2,$sim\n")
}
writer.close()

sc.stop()
