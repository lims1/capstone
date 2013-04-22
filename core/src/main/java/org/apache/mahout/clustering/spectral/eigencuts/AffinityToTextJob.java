package org.apache.mahout.clustering.spectral.eigencuts;

import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
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
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        
        // Reflection.
        job.setMapperClass(AffinityToTextMapper.class);
        job.setNumReduceTasks(0);
        
        // Launch the job.
        return job.waitForCompletion(true);
    }
    
    public static class AffinityToTextMapper extends Mapper<IntWritable, VectorWritable, Text, Text> {
        
        @Override
        public void map(IntWritable key, VectorWritable v, Context context)
                throws IOException, InterruptedException {
            // Simply convert the key and value to text formats.
            Text outKey = new Text("" + key.get());
            StringBuilder outVal = new StringBuilder();
            Vector vector = v.get();
            Iterator<Vector.Element> iter = vector.iterator();
            while (iter.hasNext()) {
                Vector.Element e = iter.next();
                outVal.append(e.get());
                if (iter.hasNext()) {
                    outVal.append(",");
                }
            }
            context.write(outKey, new Text(outVal.toString()));
        }
    }
}
