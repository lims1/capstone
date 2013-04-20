package org.apache.mahout.clustering.spectral.eigencuts;

import java.io.IOException;
import java.util.Hashtable;
import java.util.Iterator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Mapper.Context;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.MultipleInputs;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.clustering.spectral.common.VertexWritable;
import org.apache.mahout.clustering.spectral.eigencuts.AffinityCutsJob.MatrixDiagonalizeReducer;
import org.apache.mahout.clustering.spectral.eigencuts.AffinityCutsJob.affPathTaggerMapper;
import org.apache.mahout.clustering.spectral.eigencuts.AffinityCutsJob.sensitivityPathTaggerMapper;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

public final class AffinityReconstructJob {
	
	  private AffinityReconstructJob() {
	  }

	  public static void runJob(Path affPath, Path outputPath, int dimensions)
	    throws IOException, ClassNotFoundException, InterruptedException {
	    
	    // set up all the job tasks
		System.out.println("You have entered the Affinity Reconstruct Job");
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
	  
	  public static class AffinityReconstructMapper
	    extends Mapper<IntWritable, VectorWritable, IntWritable, VertexWritable> {
	    
	    @Override
	    protected void map(IntWritable key, VectorWritable row, Context context) 
	      throws IOException, InterruptedException {
	   
	    	Vector v = row.get();
	    	Iterator<Vector.Element> itr = v.iterateNonZero();
	    	while(itr.hasNext())
	    	{
	    		Vector.Element e = itr.next();
	    		VertexWritable vert = new VertexWritable(key.get(),e.index(),e.get(),null);
	    		context.write(key, vert);
	    		//don't want to write out two diagonals to the same reducer
	    		if(vert.getRow() != vert.getCol())
	    		{
	    			context.write(new IntWritable(vert.getCol()), vert);
	    		}
	    	}
	   
	    }
	  }
	  
	  public static class AffinityReconstructReducer 
	  	extends Reducer<IntWritable, VertexWritable, IntWritable, VectorWritable>{
		    @Override
		    protected void reduce(IntWritable key, Iterable<VertexWritable> vertices,
		      Context context) throws IOException, InterruptedException {
		    	
		    	int dimension = context.getConfiguration().getInt(EigencutsKeys.AFFINITY_DIMENSIONS, Integer.MAX_VALUE);
		    	RandomAccessSparseVector output = new RandomAccessSparseVector(dimension);
		    	Hashtable<Integer, VertexWritable> hash = new Hashtable<Integer, VertexWritable>(); 
		    	VertexWritable diag = new VertexWritable();
		    	for(VertexWritable v : vertices)
		    	{
		    		if(v.getRow() != key.get())
		    		{
		    			hash.put(new Integer(v.getRow()), v);
		    		}
		    		else if(v.getRow() == v.getCol())
		    		{
		    			diag = v;
		    		}
		    	}
		    	
		    	for(VertexWritable v : vertices)
		    	{
		    		//if it's not in this row, or it's the diagonal nothing needs to be done
		    		if(v.getRow() == key.get() && v.getRow() != v.getCol())
		    		{
		    			//if there's no corresponding symmetric entry, 
		    			//i.e. the symmetric entry == 0 and has been cut
		    			if(hash.get(new Integer(v.getCol())) == null)
		    			{
		    				diag.setValue(diag.getValue() + v.getValue());
		    				v.setValue(0);
		    			}
		    			
		    			output.setQuick(v.getCol(), v.getValue());
		    		}
		    	}
		    	
		    	output.setQuick(key.get(), diag.getValue());
		    	context.write(key, new VectorWritable(output));
		    }		  
	  }
}
