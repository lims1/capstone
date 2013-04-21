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
import java.util.Iterator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.MultipleInputs;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.clustering.spectral.common.VertexWritable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

/**
 * Given a matrix, this job returns a vector whose i_th element is the sum of
 * all the elements in the i_th row of the original matrix.
 */
public final class AffinityCutsJob {

	private AffinityCutsJob() {
	}

	public static void runJob(Path affPath, Path sensitivityPath,
			Path outputPath, int dimensions) throws IOException,
			ClassNotFoundException, InterruptedException {

		// set up all the job tasks
		Configuration conf = new Configuration();
		conf.setInt(EigencutsKeys.AFFINITY_DIMENSIONS, dimensions);
		Job job = new Job(conf, "AffinityCutsJob");

		job.setMapOutputKeyClass(Text.class);
		job.setMapOutputValueClass(VertexWritable.class);
		job.setOutputKeyClass(IntWritable.class);
		job.setOutputValueClass(VertexWritable.class);
		job.setOutputFormatClass(SequenceFileOutputFormat.class);
		job.setReducerClass(MatrixDiagonalizeReducer.class);

		MultipleInputs.addInputPath(job, affPath,
				SequenceFileInputFormat.class, affPathTaggerMapper.class);
		MultipleInputs.addInputPath(job, sensitivityPath,
				SequenceFileInputFormat.class,
				sensitivityPathTaggerMapper.class);

		FileOutputFormat.setOutputPath(job, outputPath);

		job.setJarByClass(AffinityCutsJob.class);

		boolean succeeded = job.waitForCompletion(true);
		if (!succeeded) {
			throw new IllegalStateException("Job failed!");
		}

	}

	public static class affPathTaggerMapper extends
			Mapper<IntWritable, VectorWritable, Text, VertexWritable> {

		@Override
		protected void map(IntWritable key, VectorWritable row, Context context)
				throws IOException, InterruptedException {

			Vector v = row.get();
			Iterator<Vector.Element> itr = v.iterateNonZero();
			while (itr.hasNext()) {
				Vector.Element e = itr.next();
				if (e.index() > key.get()) {
					String newkey = key.get() + "_" + e.index();
					context.write(new Text(newkey),
							new VertexWritable(key.get(), e.index(), e.get(),
									"aff"));
				}
			}

		}
	}

	public static class sensitivityPathTaggerMapper extends
			Mapper<IntWritable, VectorWritable, Text, VertexWritable> {

		@Override
		protected void map(IntWritable key, VectorWritable row, Context context)
				throws IOException, InterruptedException {

			Vector v = row.get();
			Iterator<Vector.Element> itr = v.iterateNonZero();
			while (itr.hasNext()) {
				Vector.Element e = itr.next();
				if (e.index() > key.get()) {
					String newkey = key.get() + "_" + e.index();
					context.write(new Text(newkey),
							new VertexWritable(key.get(), e.index(), e.get(),
									"sen"));
				}
			}
		}
	}

	public static class MatrixDiagonalizeReducer extends
			Reducer<Text, VertexWritable, IntWritable, VertexWritable> {

		@Override
		protected void reduce(Text key, Iterable<VertexWritable> values,
				Context context) throws IOException, InterruptedException {

			VertexWritable affV = null;
			VertexWritable senV = null;
			int count = 0;
			for (VertexWritable v : values) {
				if (v.getType().equals(new String("sen"))) {
					senV = v;
				} else if (v.getType().equals(new String("aff"))) {
					affV = v;
				}
				count++;
			}

			/*
			 * If there is a sensitivity value, the corresponding affinity entry
			 * gets cut. If the count is less than 2, that means either a) both
			 * the sensitivity and affinity entries were 0, b) the sensitivity
			 * was nonzero, but the affinity was zero (i.e. already cut), or c)
			 * the affinity was nonzero, but the sensitivity was zero (i.e. it
			 * didn't meet the threshold or was suppressed). In any of these
			 * cases, nothing further needs to be done.
			 */
			if (count == 2) {
				if (senV == null) {
					throw new NullPointerException();
				} else {
					context.write(new IntWritable(affV.getRow()), affV);
				}
			}
		}
	}
}
