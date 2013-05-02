package org.apache.mahout.clustering.spectral.eigencuts;

import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

/**
 * Quick and dirty method for pulling out the final affinity matrix into
 * a readable format. Should output each row as a comma-separated list
 * of all N values in the row vector, including 0s.
 */
public final class AffinityToTextJob {

    public static boolean runJob(Path affinityMatrix, Path output)
            throws IOException, ClassNotFoundException, InterruptedException {
        Configuration conf = new Configuration();
        Job job = new Job(conf, "AffinityToTextJob - Final Job");
        job.setJarByClass(AffinityToTextJob.class);
        
        // Data formats.
        job.setInputFormatClass(SequenceFileInputFormat.class);
        job.setOutputFormatClass(TextOutputFormat.class);
        FileInputFormat.addInputPath(job, affinityMatrix);
        FileOutputFormat.setOutputPath(job, output);
        
        // Job formats.
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);
        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(Text.class);
        
        // Reflection.
        job.setMapperClass(AffinityToTextMapper.class);
        job.setNumReduceTasks(0);
        
        // Launch the job.
        return job.waitForCompletion(true);
    }
    
    public static class AffinityToTextMapper extends Mapper<IntWritable, VectorWritable, NullWritable, Text> {
        
        @Override
        public void map(IntWritable key, VectorWritable v, Context context)
                throws IOException, InterruptedException {
            // Output format same as input format: row,col,val
            Iterator<Vector.Element> iter = v.get().iterateNonZero();
            while (iter.hasNext()) {
                Vector.Element e = iter.next();
                String outValue = String.format("%s,%s,%s", key.get(), e.index(), e.get());
                context.write(NullWritable.get(), new Text(outValue));
            }
        }
    }
}
