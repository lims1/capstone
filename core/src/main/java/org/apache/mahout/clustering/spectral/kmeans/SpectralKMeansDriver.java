/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.clustering.spectral.kmeans;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.kmeans.KMeansDriver;
import org.apache.mahout.clustering.kmeans.RandomSeedGenerator;
import org.apache.mahout.clustering.spectral.common.AffinityMatrixInputJob;
import org.apache.mahout.clustering.spectral.common.MatrixDiagonalizeJob;
import org.apache.mahout.clustering.spectral.common.UnitVectorizerJob;
import org.apache.mahout.clustering.spectral.common.VectorMatrixMultiplicationJob;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.ClassUtils;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.common.Pair;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.decomposer.lanczos.LanczosState;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.apache.mahout.math.hadoop.decomposer.DistributedLanczosSolver;
import org.apache.mahout.math.hadoop.decomposer.EigenVerificationJob;
import org.apache.mahout.math.hadoop.stochasticsvd.SSVDSolver; 
import org.apache.mahout.math.VectorWritable;

import java.io.*;
import java.io.IOException;
import java.util.List;
import java.util.Map;


/**
 * Implementation of the EigenCuts spectral clustering algorithm.
 * This implementation is for testing and debugging. 
 * 
 * Using the variables below the user can:
 * 		select to use either SSVDSolver or DistributedLanczosSolver for the Eigen decomposition. 
 * 		change the number of iterations in SSVD
 * 		choose whether to keep the temp files that are created during a job
 * 		have the output printed to a text file 
 * 
 * All of the steps involved in testing have timers built around them and the result is printed at
 * the top of the output text file. 
 * 
 * See the README file for a description of the algorithm, testing results, and other details.
 */
public class SpectralKMeansTester extends AbstractJob {

  public static final double OVERSHOOT_MULTIPLIER = 2.0;

  
  public static final boolean USE_SSVD = true;
  public static final boolean OUTPUT_TO_TEXTFile = true;
  public static final boolean DELETE_TEMP_FILES = true;  
  
  static long ssvdTime = -1;
  static long lanczosTime = -1;
  static long normalizeTime = -1; 
  static int ssvdIterations = 0;

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new SpectralKMeansTester(), args);
  }

  @Override
  public int run(String[] arg0)
    throws IOException, ClassNotFoundException, InstantiationException, IllegalAccessException, InterruptedException {
    
    Configuration conf = getConf();
    addInputOption();
    addOutputOption();
    addOption("dimensions", "d", "Square dimensions of affinity matrix", true);
    addOption("clusters", "k", "Number of clusters and top eigenvectors", true);
    addOption(DefaultOptionCreator.distanceMeasureOption().create());
    addOption(DefaultOptionCreator.convergenceOption().create());
    addOption(DefaultOptionCreator.maxIterationsOption().create());
    addOption(DefaultOptionCreator.overwriteOption().create());

    Map<String, List<String>> parsedArgs = parseArguments(arg0);
    if (parsedArgs == null) {
      return 0;
    }

    Path input = getInputPath();
    Path output = getOutputPath();
    if (hasOption(DefaultOptionCreator.OVERWRITE_OPTION)) {
    	HadoopUtil.delete(conf, output); }
    int numDims = Integer.parseInt(getOption("dimensions"));
    int clusters = Integer.parseInt(getOption("clusters"));
    String measureClass = getOption(DefaultOptionCreator.DISTANCE_MEASURE_OPTION);
    DistanceMeasure measure = ClassUtils.instantiateAs(measureClass, DistanceMeasure.class);
    double convergenceDelta = Double.parseDouble(getOption(DefaultOptionCreator.CONVERGENCE_DELTA_OPTION));
    int maxIterations = Integer.parseInt(getOption(DefaultOptionCreator.MAX_ITERATIONS_OPTION));

    run(conf, input, output, numDims, clusters, measure, convergenceDelta, maxIterations);

    return 0;
  }

  /**
   * Run the Spectral KMeans clustering on the supplied arguments
   * 
   * @param conf the Configuration to be used
   * @param input the Path to the input tuples directory
   * @param output the Path to the output directory
   * @param numDims the int number of dimensions of the affinity matrix
   * @param clusters the int number of eigenvectors and thus clusters to produce
   * @param measure the DistanceMeasure for the k-Means calculations
   * @param convergenceDelta the double convergence delta for the k-Means calculations
   * @param maxIterations the int maximum number of iterations for the k-Means calculations
   */
  public static void run(
		  Configuration conf,
          Path input,
          Path output,
          int numDims,
          int clusters,
          DistanceMeasure measure,
          double convergenceDelta,
          int maxIterations)
    throws IOException, InterruptedException, ClassNotFoundException {
    
    Path outputCalc = new Path(output, "calculations");
    Path outputTmp = new Path(output, "temporary");

    // Take in the raw CSV text file and split it ourselves,
    // creating our own SequenceFiles for the matrices to read later 
    // (similar to the style of syntheticcontrol.canopy.InputMapper)
    Path affSeqFiles = new Path(outputCalc, "seqfile");
    AffinityMatrixInputJob.runJob(input, affSeqFiles, numDims, numDims);

    // Construct the affinity matrix using the newly-created sequence files
    DistributedRowMatrix A = 
    		new DistributedRowMatrix(affSeqFiles, new Path(outputTmp, "afftmp"), numDims, numDims); 

    Configuration depConf = new Configuration(conf);
    A.setConf(depConf);
    
    // Construct the diagonal matrix D (represented as a vector)
    Vector D = MatrixDiagonalizeJob.runJob(affSeqFiles, numDims);
    
    //Calculate the normalized Laplacian of the form: L = D^(-0.5)AD^(-0.5)
    DistributedRowMatrix L = VectorMatrixMultiplicationJob.runJob(affSeqFiles, D,
    		new Path(outputCalc, "laplacian-" + (System.nanoTime() & 0xFF)), new Path(outputCalc, outputCalc));
   	L.setConf(depConf);
    
    Path data;

   if(USE_SSVD){ // Otherwise uses slower Lanczos
	   
	// SSVD requires an array of Paths to function. So we pass in an array of length one
    Path[] LPath = new Path[1];
   	LPath[0] = L.getRowPath();

   	
   	Path SSVDout = new Path(outputCalc, "SSVD");
   	SSVDSolver solveIt = new SSVDSolver(depConf, 
   										LPath, 
   										SSVDout, 
   										1000, // Vertical height of a q-block
   										clusters, 
   										15, // Oversampling 
   										100); // # of reduce tasks
  	solveIt.setComputeV(false); 
  	solveIt.setComputeU(true);
  	solveIt.setOverwrite(true);
  	solveIt.setQ(ssvdIterations); // Set the power iterations (0 was fastest most accurate in testing)
  	solveIt.setBroadcast(false); 
  	// setBroadcast should be set to true is running on a distributed system, but on a single
  	// machine it must be set to false. The documentation says that the default is false but 
  	// the default is actually true. 

  	ssvdTime = System.currentTimeMillis();
  	solveIt.run();
  	ssvdTime = System.currentTimeMillis() - ssvdTime;
  	data = new Path(solveIt.getUPath()); // Needs "new Path", getUPath method returns a String
  	
   }else{

	    // Perform eigen-decomposition using LanczosSolver
	    // since some of the eigen-output is spurious and will be eliminated
	    // upon verification, we have to aim to overshoot and then discard
	    // unnecessary vectors later
	    int overshoot = (int) ((double) clusters * OVERSHOOT_MULTIPLIER);
	    DistributedLanczosSolver solver = new DistributedLanczosSolver();
	    LanczosState state = new LanczosState(L, overshoot, solver.getInitialVector(L));
	    Path lanczosSeqFiles = new Path(outputCalc, "eigenvectors-" + (System.nanoTime() & 0xFF));
	    lanczosTime = System.currentTimeMillis();
	    solver.runJob(conf,
	                  state,
	                  overshoot,
	                  true,
	                  lanczosSeqFiles.toString());

	    // perform a verification
	    EigenVerificationJob verifier = new EigenVerificationJob();
	    Path verifiedEigensPath = new Path(outputCalc, "eigenverifier");
	    verifier.runJob(conf, 
	    				lanczosSeqFiles, 
	    				L.getRowPath(), 
	    				verifiedEigensPath, 
	    				true, 
	    				1.0, 
	    				clusters);
	    lanczosTime = System.currentTimeMillis() - lanczosTime;
	    Path cleanedEigens = verifier.getCleanedEigensPath();
	    DistributedRowMatrix W = new DistributedRowMatrix(cleanedEigens, new Path(cleanedEigens, "tmp"), clusters, numDims);
	    W.setConf(depConf);
	    DistributedRowMatrix Wtrans = W.transpose();
	    data = Wtrans.getRowPath();
   }
   	

   		// Normalize the rows of Wt to unit length
   		// normalize is important because it reduces the occurrence of two unique clusters  combining into one 
	   Path unitVectors = new Path(outputCalc, "unitvectors-" + (System.nanoTime() & 0xFF));
	   normalizeTime = System.currentTimeMillis();
	   UnitVectorizerJob.runJob(data, unitVectors);
	   normalizeTime = System.currentTimeMillis() - normalizeTime;
	   DistributedRowMatrix Wt = new DistributedRowMatrix(unitVectors, new Path(unitVectors, "tmp"), clusters, numDims);
	   Wt.setConf(depConf);
	   data = Wt.getRowPath();


    
    // Generate random initial clusters
    Path initialclusters = RandomSeedGenerator.buildRandom(conf, data,
    		new Path(output, Cluster.INITIAL_CLUSTERS_DIR), clusters, measure);
   
    // Run the KMeansDriver
    Path answer = new Path(output, "kmeans_out-" + (System.nanoTime() & 0xFF));
	KMeansDriver.run(conf, data, initialclusters, answer,
	    		measure,convergenceDelta, maxIterations, true, 0.0, false);
 
	if(DELETE_TEMP_FILES){
		HadoopUtil.delete(conf, outputCalc);
		
	}
	
	
	// Prints out results into a text file. Format of three info lines that begin with >
	// The rest of the lines are point_number,cluster_number 
	// For it to run the FileWriter path should be changed according to your preference
	if(OUTPUT_TO_TEXTFile){   
		// Read through the cluster assignments
	    Path clusteredPointsPath = new Path(answer, "clusteredPoints");
	    Path inputPath = new Path(clusteredPointsPath, "part-m-00000");
	    
	    FileWriter fstream = new FileWriter("result" + (System.nanoTime() & 0xFF) + ".txt");
	    BufferedWriter out = new BufferedWriter(fstream);

	    int id = 0;
	    try{
	    	out.write(">  " + input.toString() +  "\n");
	    	out.write(">  " +  "ssvdTime = "  + ssvdTime + " lanczosTime = " + lanczosTime 
	    				+ " normalizeTime = " + normalizeTime +  "\n");
	    	out.write(">  " + "ssvdIterations = " + ssvdIterations + "  total: " + (ssvdTime + normalizeTime) +  "\n");
	    	
	    for (Pair<IntWritable,VectorWritable> record 
	         : new SequenceFileIterable<IntWritable, VectorWritable>(inputPath, new Configuration())) {

	    	out.write("" + id + "," + record.getFirst().get() + "\n");
	    	id++;
	    	}
	    }catch (Exception e){
		     System.out.println("IO ERROR");		    
	    
	    } out.close();
	}
    
  }
  }



////////////////////////  End of File  //////////////////////////////////////////////////