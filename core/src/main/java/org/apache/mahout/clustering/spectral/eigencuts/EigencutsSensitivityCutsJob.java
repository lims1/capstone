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

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.MultipleInputs;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.clustering.spectral.common.EigencutsVectorCache;
import org.apache.mahout.math.MultiLabelVectorWritable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

/**
 * 
 * This job takes in the affinity matrix and the eigenvectors. It creates
 * an n x n sensitivity matrix of the most negative sensitivities calculated from
 * the eigenvectors that pass the threshold (-log(2)/log(eigenvalue)). The reducer
 * performs the necessary cuts to the affinity matrix, while storing the cut values
 * in the diagonals. This affinity matrix is not complete and should be followed up by
 * another map reduce job 
 * 
 * <p>
 * Overall, this creates an n-by-n (possibly sparse) affinity matrix with a maximum of
 * n^2 non-zero elements, minimum of n non-zero elements.
 * </p>
 */
public final class EigencutsSensitivityCutsJob {

	private EigencutsSensitivityCutsJob() {
	}
	 
	enum CUTSCOUNTER {
		    NUM_CUTS
		  }

	/**
	 * Initializes the configuration tasks, loads the needed data into the HDFS
	 * cache, and executes the job.
	 * 
	 * @param eigenvalues
	 *            Vector of eigenvalues
	 * @param diagonal
	 *            Vector representing the diagonal matrix
	 * @param affinityPath
	 * 			  Path to the affinity matrix 
	 * @param eigenvectors
	 *            Path to the DRM of eigenvectors
	 * @param output
	 *            Path to the output matrix (will have between n and full-rank
	 *            non-zero elements)
	 * @return 
	 */
	public static long runJob(Vector eigenvalues, Vector diagonal,
			Path affinityPath, Path eigenvectors, double beta, double tau,
			double delta, double epsilon, Path output) throws IOException,
			ClassNotFoundException, InterruptedException {

		// save the two vectors to the distributed cache
		Configuration jobConfig = new Configuration();
		Path eigenOutputPath = new Path(output.getParent(), "eigenvalues");
		Path diagOutputPath = new Path(output.getParent(), "diagonal");
		jobConfig.set(EigencutsKeys.VECTOR_CACHE_BASE, output.getParent()
				.getName());
		EigencutsVectorCache.save(new IntWritable(
				EigencutsKeys.EIGENVALUES_CACHE_INDEX), eigenvalues,
				eigenOutputPath, new IntWritable(
						EigencutsKeys.DIAGONAL_CACHE_INDEX), diagonal,
				diagOutputPath, jobConfig);
		// set up the rest of the job
		jobConfig.set(EigencutsKeys.BETA, Double.toString(beta));
		System.out.println("Set beta:" + Double.toString(beta));
		jobConfig.set(EigencutsKeys.EPSILON, Double.toString(epsilon));
		System.out.println("Set epsilon:" + Double.toString(epsilon));
		jobConfig.set(EigencutsKeys.DELTA, Double.toString(delta));
		System.out.println("Set delta:" + Double.toString(delta));
		jobConfig.set(EigencutsKeys.TAU, Double.toString(tau));
		System.out.println("Set tau:" + Double.toString(tau));

		Job job = new Job(jobConfig, "EigencutsSensitivityJob");
		job.setJarByClass(EigencutsSensitivityCutsJob.class);

		// Affinity matrix read in as a sequence file and tagged with
		// affPathTaggerMapper.class
		MultipleInputs.addInputPath(job, affinityPath,
				SequenceFileInputFormat.class, affPathTaggerMapper.class);

		// Sensitivity matrix read in as a sequence file and tagged with
		// sensitivityPathTaggerMapper.class
		MultipleInputs.addInputPath(job, eigenvectors,
				SequenceFileInputFormat.class,
				EigencutsSensitivityCutsMapper.class);

		job.setMapOutputKeyClass(IntWritable.class);
		job.setMapOutputValueClass(MultiLabelVectorWritable.class);
		job.setOutputKeyClass(IntWritable.class);
		job.setOutputValueClass(VectorWritable.class);
		job.setOutputFormatClass(SequenceFileOutputFormat.class);

		job.setReducerClass(EigencutsSensitivityCutsReducer.class);

		FileOutputFormat.setOutputPath(job, output);
		System.out.println("Sensitivity output:" + output);

		boolean succeeded = job.waitForCompletion(true);
		if (!succeeded) {
			throw new IllegalStateException("Job failed!");
		}
		
		return job.getCounters().findCounter(CUTSCOUNTER.NUM_CUTS).getValue();
	}

	public static class affPathTaggerMapper
			extends
			Mapper<IntWritable, VectorWritable, IntWritable, MultiLabelVectorWritable> {

		@Override
		protected void map(IntWritable key, VectorWritable row, Context context)
				throws IOException, InterruptedException {

			System.out.println("You are in rowPathTaggerMapper");
			System.out.println("Key:" + key);
			System.out.println("Row:" + row.get());

			// Tag the vectors coming in from affinity matrix as 0
			int[] i = { 0 };
			MultiLabelVectorWritable node = new MultiLabelVectorWritable(
					row.get(), i);

			context.write(key, node);

		}

	}

}
