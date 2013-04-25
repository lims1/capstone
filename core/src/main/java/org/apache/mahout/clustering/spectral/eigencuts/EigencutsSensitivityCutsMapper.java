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
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.clustering.spectral.common.EigencutsVectorCache;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.MultiLabelVectorWritable;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.Functions;

public class EigencutsSensitivityCutsMapper
		extends
		Mapper<IntWritable, VectorWritable, IntWritable, MultiLabelVectorWritable> {

	private Vector eigenvalues;
	private Vector diagonal;
	private double beta0;
	private double epsilon;

	@Override
	protected void setup(Context context) throws IOException,
			InterruptedException {
		super.setup(context);
		Configuration config = context.getConfiguration();
		beta0 = Double.parseDouble(config.get(EigencutsKeys.BETA));
		epsilon = Double.parseDouble(config.get(EigencutsKeys.EPSILON));
		// read in the two vectors from the cache
		eigenvalues = EigencutsVectorCache.load(
				EigencutsKeys.EIGENVALUES_CACHE_INDEX, config);
		diagonal = EigencutsVectorCache.load(
				EigencutsKeys.DIAGONAL_CACHE_INDEX, config);
		if (!(eigenvalues instanceof SequentialAccessSparseVector || eigenvalues instanceof DenseVector)) {
			eigenvalues = new SequentialAccessSparseVector(eigenvalues);
		}
		if (!(diagonal instanceof SequentialAccessSparseVector || diagonal instanceof DenseVector)) {
			diagonal = new SequentialAccessSparseVector(diagonal);
		}
	}

	@Override
	protected void map(IntWritable row, VectorWritable vector, Context context)
			throws IOException, InterruptedException {

		System.out.println("Row: " + row.get());
		// first, does this particular eigenvector even pass the required
		// threshold?
		double eigenvalue = Math.abs(eigenvalues.get(row.get())); //should be -1?
		double betak = -Functions.LOGARITHM.apply(2)
				/ Functions.LOGARITHM.apply(eigenvalue);
		if (eigenvalue >= 1.0 || betak <= epsilon * beta0) {
			// doesn't pass the threshold! quit
			return;
		}
		// go through the vector, performing the calculations
		// sadly, no way to get around n^2 computations
		// for simplicity purposes non-maximal suppression also not included for
		// now
		Vector ev = vector.get();
		Configuration conf = context.getConfiguration();
		double threshold = Double.parseDouble(conf.get(EigencutsKeys.TAU))
				/ Double.parseDouble(conf.get(EigencutsKeys.DELTA));

		SequentialAccessSparseVector v;
		for (int i = 0; i < ev.size(); i++) {

			v = new SequentialAccessSparseVector(conf.getInt(
					EigencutsKeys.AFFINITY_DIMENSIONS, Integer.MAX_VALUE), 100);

			for (int j = 0; j < ev.size(); j++) {
				double sij = performSensitivityCalculation(eigenvalue,
						ev.get(i), ev.get(j), diagonal.get(i), diagonal.get(j));

				if (sij < threshold) {
					v.setQuick(j, sij);
				}
			}
			// Tag the vectors coming in to this mapper as 1 for label
			int[] k = { 1 };
			MultiLabelVectorWritable node = new MultiLabelVectorWritable(v, k);
			context.write(new IntWritable(i), node);
		}
	}

	/**
	 * Helper method, performs the actual calculation. Looks something like
	 * this:
	 * 
	 * (log(2) / lambda_k * log(lambda_k) * log(lambda_k^beta0 / 2)) * [ -
	 * (((u_i / sqrt(d_i)) - (u_j / sqrt(d_j)))^2 + (1 - lambda) * ((u_i^2 /
	 * d_i) + (u_j^2 / d_j))) ]
	 */
	private double performSensitivityCalculation(double eigenvalue, double evi,
			double evj, double diagi, double diagj) {

		double firsthalf = Functions.LOGARITHM.apply(2)
				/ (eigenvalue * Functions.LOGARITHM.apply(eigenvalue) * Functions.LOGARITHM
						.apply(Functions.POW.apply(eigenvalue, beta0) / 2));

		double secondhalf = -Functions.POW.apply(
				evi / Functions.SQRT.apply(diagi) - evj
						/ Functions.SQRT.apply(diagj), 2)
				+ (1.0 - eigenvalue)
				* (Functions.POW.apply(evi, 2) / diagi + Functions.POW.apply(
						evj, 2) / diagj);

		return firsthalf * secondhalf;
	}

	/**
	 * Utility helper method, used for unit testing.
	 */
	void setup(double beta0, double epsilon, Vector eigenvalues, Vector diagonal) {
		this.beta0 = beta0;
		this.epsilon = epsilon;
		this.eigenvalues = eigenvalues;
		this.diagonal = diagonal;
	}
}
