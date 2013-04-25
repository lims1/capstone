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

package org.apache.mahout.clustering.spectral.eigencuts;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.clustering.spectral.common.AffinityMatrixInputJob;
import org.apache.mahout.clustering.spectral.common.MatrixDiagonalizeJob;
import org.apache.mahout.clustering.spectral.common.UnitVectorizerJob;
import org.apache.mahout.clustering.spectral.common.VectorMatrixMultiplicationJob;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.decomposer.lanczos.LanczosState;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.apache.mahout.math.hadoop.decomposer.DistributedLanczosSolver;
import org.apache.mahout.math.hadoop.decomposer.EigenVerificationJob;
import org.apache.mahout.math.stats.OnlineSummarizer;

public class EigencutsDriver extends AbstractJob {

	public static final double EPSILON_DEFAULT = 0.25;
	public static final double TAU_DEFAULT = -0.1;
	public static final double OVERSHOOT_MULTIPLIER = 1.5;
	public static final int REDUCERS = 10;
	public static final int BLOCKHEIGHT = 30000;
	public static final int OVERSAMPLING = 15;
	public static final int POWERITERS = 0;
	public static final int CUTSITERS = 1;

	public static void main(String[] args) throws Exception {
		ToolRunner.run(new EigencutsDriver(), args);
	}

	@Override
	public int run(String[] arg0) throws Exception {

		// set up command line arguments
		Configuration conf = getConf();
		addInputOption();
		addOutputOption();
		addOption("half-life", "b", "Minimal half-life threshold", true);
		addOption("dimensions", "d", "Square dimensions of affinity matrix",
				true);
		addOption("epsilon", "e", "Half-life threshold coefficient",
				Double.toString(EPSILON_DEFAULT));
		addOption("tau", "t", "Threshold for cutting affinities",
				Double.toString(TAU_DEFAULT));
		addOption("eigenrank", "k", "Number of top eigenvectors to use", true);
		addOption(DefaultOptionCreator.overwriteOption().create());
		addOption("reduceTasks", "t", "Number of reducers for SSVD",
				String.valueOf(REDUCERS));
		addOption("outerProdBlockHeight", "oh",
				"Block height of outer products for SSVD",
				String.valueOf(BLOCKHEIGHT));
		addOption("oversampling", "p", "Oversampling parameter for SSVD",
				String.valueOf(OVERSAMPLING));
		addOption("powerIter", "q", "Additional power iterations for SSVD",
				String.valueOf(POWERITERS));
		addOption(
				"cutsIter",
				"c",
				"Constraint that only iterates x number of times (defaults to 1), maximum",
				String.valueOf(CUTSITERS));

		Map<String, List<String>> parsedArgs = parseArguments(arg0);
		if (parsedArgs == null) {
			return 0;
		}

		// read in the command line values
		Path input = getInputPath();
		Path output = getOutputPath();
		if (hasOption(DefaultOptionCreator.OVERWRITE_OPTION)) {
			HadoopUtil.delete(getConf(), output);
		}

		int numDims = Integer.parseInt(getOption("dimensions"));
		double halflife = Double.parseDouble(getOption("half-life"));
		double epsilon = Double.parseDouble(getOption("epsilon"));
		double tau = Double.parseDouble(getOption("tau"));
		int eigenrank = Integer.parseInt(getOption("eigenrank"));
		int reducers = Integer.parseInt(getOption("reduceTasks"));
		int blockheight = Integer.parseInt(getOption("outerProdBlockHeight"));
		int oversampling = Integer.parseInt(getOption("oversampling"));
		int poweriters = Integer.parseInt(getOption("powerIter"));
		int cutsiters = Integer.parseInt(getOption("cutsIter"));

		run(conf, input, output, numDims, eigenrank, halflife, epsilon, tau,
				reducers, blockheight, oversampling, poweriters, cutsiters);

		return 0;
	}

	/**
	 * Run the Eigencuts clustering algorithm using the supplied arguments
	 * 
	 * @param conf
	 *            the Configuration to use
	 * @param input
	 *            the Path to the directory containing input affinity tuples
	 * @param output
	 *            the Path to the output directory
	 * @param eigenrank
	 *            The number of top eigenvectors/eigenvalues to use
	 * @param numDims
	 *            the int number of dimensions of the square affinity matrix
	 * @param halflife
	 *            the double minimum half-life threshold
	 * @param epsilon
	 *            the double coefficient for setting minimum half-life threshold
	 * @param tau
	 *            the double tau threshold for cutting links in the affinity
	 *            graph
	 */
	public static void run(Configuration conf, Path input, Path output,
			int numDims, int eigenrank, double halflife, double epsilon,
			double tau, int numReducers, int blockHeight, int oversampling,
			int poweriters, int cutiters) throws IOException,
			InterruptedException, ClassNotFoundException {
		// set the instance variables
		// create a few new Paths for temp files and transformations
		Path outputCalc = new Path(output, "calculations");
		Path outputTmp = new Path(output, "temporary");

		// Take in the raw CSV text file and split it ourselves,
		// creating our own SequenceFiles for the matrices to read later
		// (similar to the style of syntheticcontrol.canopy.InputMapper)
		Path affSeqFiles = new Path(outputCalc, "seqfile_1");
		AffinityMatrixInputJob.runJob(input, affSeqFiles, numDims, numDims);

		// Construct the affinity matrix using the newly-created sequence files
		DistributedRowMatrix A = new DistributedRowMatrix(affSeqFiles,
				new Path(outputTmp, "afftmp"), numDims, numDims);

		A.setConf(conf);
		
		// Construct the diagonal matrix D (represented as a vector)
		Vector D = MatrixDiagonalizeJob.runJob(affSeqFiles, numDims);

		long numCuts = 0;
		int iterations = 0;
		Path data;

		do {
			iterations++;

			// first three steps are the same as spectral k-means:
			// 1) calculate D from A 
			// 2) calculate L = D^-0.5 * A * D^-0.5
			// 3) calculate eigenvectors of L
			
			// Calculate the normalized Laplacian of the form: L =
			// D^(-0.5)AD^(-0.5)
			Path laplacian = new Path(outputCalc, "laplacian");
			DistributedRowMatrix L = VectorMatrixMultiplicationJob.runJob(
					A.getRowPath(), D, new Path(outputCalc, "laplacian"), new Path(outputCalc, outputCalc) );
			L.setConf(conf);
			
			int overshoot = Math.min((int) ((double) eigenrank * OVERSHOOT_MULTIPLIER), numDims);
			DistributedLanczosSolver solver = new DistributedLanczosSolver();
			LanczosState state = new LanczosState(L, overshoot, solver.getInitialVector(L));
			Path lanczosSeqFiles = new Path(outputCalc, "eigenvectors");
			
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
							overshoot);
			
			Path cleanedEigens = verifier.getCleanedEigensPath();
			DistributedRowMatrix W = new DistributedRowMatrix(
					cleanedEigens, new Path(cleanedEigens, "tmp"), overshoot, numDims);
			W.setConf(conf);

			data = W.getRowPath();
		
		// Normalize the rows of Wt to unit length
		// normalize is important because it reduces the occurrence of two unique clusters  combining into one 
		Path unitVectors = new Path(outputCalc, "unitvectors");
		
		UnitVectorizerJob.runJob(data, unitVectors, overshoot);
		
		DistributedRowMatrix Wt = new DistributedRowMatrix(
				unitVectors, new Path(unitVectors, "tmp"), eigenrank, numDims);
		Wt.setConf(conf);
		data = Wt.getRowPath();
		
		    ArrayList<Double> eigenValues = new ArrayList<Double>(eigenrank);
		      for (int i=overshoot-1; i>=0; i--) { 
		        eigenValues.add(state.getSingularValue(i));
		      }
		   
		    System.out.println("===EIGENVALUES===");
		    System.out.println(eigenValues);
		    
		    Vector evs = listToVector(eigenValues);

			// calculate sensitivities (step 4 and step 5)

			Path asymmetricA = new Path(outputCalc, "asymmetricA");

			// Need a way to check to see if there are any sensitivities in the
			// sensitivity matrix at all
			numCuts = EigencutsSensitivityCutsJob.runJob(evs, D,
					A.getRowPath(), data, halflife, tau, median(D),
					epsilon, asymmetricA);

			System.out.println("Number of cuts:" + numCuts);

			if (numCuts > 0) {

				
				Path affSeqFileTmp = new Path(outputCalc, "seqfile_" + (iterations+1));

				AffinityReconstructJob.runJob(asymmetricA, affSeqFileTmp,
						numDims);

				A = new DistributedRowMatrix(affSeqFileTmp, new Path(outputCalc,
						"afftmp_" + (iterations+1)), numDims, numDims);
				A.setConf(conf);
				
				//Remove previous seqfile
				HadoopUtil.delete(conf, affSeqFiles);
				//Remove previous laplacian matrix
				HadoopUtil.delete(conf, laplacian);
				//Remove previous sensitivities
				HadoopUtil.delete(conf,asymmetricA);
				//Remove previous unitvectors
				HadoopUtil.delete(conf, unitVectors);
				affSeqFiles = affSeqFileTmp;
			}

		} while (numCuts > 0 && iterations < cutiters);

	    // Run a hack job to convert the affinity matrix to a text file.
        AffinityToTextJob.runJob(A.getRowPath(), new Path(outputCalc.getParent(), "finalOutput"));
	}

	/**
	 * Does most of the heavy lifting in setting up Paths, configuring return
	 * values, and generally performing the tedious administrative tasks
	 * involved in an eigen-decomposition and running the verifier
	 */
	public static DistributedRowMatrix performEigenDecomposition(
			Configuration conf, DistributedRowMatrix input, LanczosState state,
			int numEigenVectors, int overshoot, Path tmp) throws IOException {

		DistributedLanczosSolver solver = new DistributedLanczosSolver();
		Path seqFiles = new Path(tmp, "eigendecomp");
		solver.runJob(conf, state, overshoot, true, seqFiles.toString());

		// now run the verifier to trim down the number of eigenvectors
		EigenVerificationJob verifier = new EigenVerificationJob();
		Path verifiedEigens = new Path(tmp, "verifiedeigens");
		verifier.runJob(conf, seqFiles, input.getRowPath(), verifiedEigens,
				false, 1.0, numEigenVectors);
		Path cleanedEigens = verifier.getCleanedEigensPath();
		return new DistributedRowMatrix(cleanedEigens, new Path(cleanedEigens,
				"tmp"), numEigenVectors, input.numRows());
	}

	/**
	 * A quick and dirty hack to compute the median of a vector...
	 * 
	 * @param v
	 * @return
	 */
	private static double median(Vector v) {
		OnlineSummarizer med = new OnlineSummarizer();
		if (v.size() < 100) {
			return v.zSum() / v.size();
		}
		for (Vector.Element e : v) {
			med.add(e.get());
		}
		return med.getMedian();
	}
	
	 private static Vector listToVector(Collection<Double> list) {
		    Vector retval = new DenseVector(list.size());
		    int index = 0;
		    for (Double d : list) {
		      retval.setQuick(index++, d);
		    }
		    return retval;
		  }
	
}