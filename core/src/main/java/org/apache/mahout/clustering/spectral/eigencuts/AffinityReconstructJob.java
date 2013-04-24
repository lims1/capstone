package org.apache.mahout.clustering.spectral.eigencuts;

import java.io.IOException;
import java.util.Hashtable;
import java.util.Iterator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.clustering.spectral.common.VertexWritable;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

public final class AffinityReconstructJob {

	private AffinityReconstructJob() {
	}

	public static void runJob(Path affPath, Path outputPath, int dimensions)
			throws IOException, ClassNotFoundException, InterruptedException {

		// set up all the job tasks
		Configuration conf = new Configuration();
		conf.setInt(EigencutsKeys.AFFINITY_DIMENSIONS, dimensions);

		Job job = new Job(conf, "AffinityReconstructJob");
		job.setJarByClass(AffinityReconstructJob.class);
		job.setInputFormatClass(SequenceFileInputFormat.class);
		job.setMapOutputKeyClass(IntWritable.class);
		job.setMapOutputValueClass(VertexWritable.class);
		job.setOutputKeyClass(IntWritable.class);
		job.setOutputValueClass(VectorWritable.class);
		job.setOutputFormatClass(SequenceFileOutputFormat.class);
		job.setMapperClass(AffinityReconstructMapper.class);
		job.setReducerClass(AffinityReconstructReducer.class);

		FileInputFormat.addInputPath(job, affPath);
		FileOutputFormat.setOutputPath(job, outputPath);

		boolean succeeded = job.waitForCompletion(true);
		if (!succeeded) {
			throw new IllegalStateException("Job failed!");
		}
	}

	public static class AffinityReconstructMapper extends
			Mapper<IntWritable, VectorWritable, IntWritable, VertexWritable> {

		@Override
		protected void map(IntWritable key, VectorWritable row, Context context)
				throws IOException, InterruptedException {

			Vector v = row.get();
			Iterator<Vector.Element> itr = v.iterateNonZero();
			while (itr.hasNext()) {
				Vector.Element e = itr.next();
				VertexWritable vert = new VertexWritable(key.get(), e.index(),
						e.get(), "");
				context.write(key, vert);
				// don't want to write out two diagonals to the same reducer
				if (vert.getRow() != vert.getCol()) {
					context.write(new IntWritable(vert.getCol()), vert);
				}
			}

		}
	}

	public static class AffinityReconstructReducer extends
			Reducer<IntWritable, VertexWritable, IntWritable, VectorWritable> {
		@Override
		protected void reduce(IntWritable key,
				Iterable<VertexWritable> vertices, Context context)
				throws IOException, InterruptedException {
			System.out.println("In AffinityReconstructReducer " + key.get());
			int dimension = context.getConfiguration().getInt(
					EigencutsKeys.AFFINITY_DIMENSIONS, Integer.MAX_VALUE);
			RandomAccessSparseVector output = new RandomAccessSparseVector(
					dimension);
			Hashtable<Integer, VertexWritable> hash = new Hashtable<Integer, VertexWritable>();
			for (VertexWritable v : vertices) {
				if (v.getRow() != key.get()) {
					hash.put(new Integer(v.getRow()), new VertexWritable(v.getRow(),v.getCol(),v.getValue(),""));
				} else {
					output.setQuick(v.getCol(), v.getValue());
				}
			}
			
			Iterator<Vector.Element> iter = output.iterateNonZero();

			while (iter.hasNext()) {
				Vector.Element v = iter.next();
				// if it's not in this row, or it's the diagonal nothing needs
				// to be done
				if (v.index() != key.get()) {
					// if there's no corresponding symmetric entry,
					// i.e. the symmetric entry == 0 and has been cut
					if (hash.get(new Integer(v.index())) == null) {
						System.out.println("Cutting (row,col,val): (" + key.get() + "," + v.index() + "," + v.get() + ")");
						output.setQuick(key.get(),output.get(key.get()) + v.get());
						output.setQuick(v.index(),0);
					}

				}
			}
			
			System.out.println("Output row: " + output);
			context.write(key, new VectorWritable(output));
		}
	}
}
