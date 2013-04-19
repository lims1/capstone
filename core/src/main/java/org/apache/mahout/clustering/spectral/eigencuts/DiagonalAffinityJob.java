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
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.MultipleInputs;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.clustering.spectral.eigencuts.EigencutsAffinityCutsJob.CUTSCOUNTER;
import org.apache.mahout.math.MultiLabelVectorWritable;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

/**
 * Given a matrix, this job returns a vector whose i_th element is the 
 * sum of all the elements in the i_th row of the original matrix.
 */
public final class DiagonalAffinityJob {

  private DiagonalAffinityJob() {
  }
  
  /**
   * 
   * @param affPath path of the affinity matrix
   * @param sensitivityPath path to the sensitivityMatrix
   * @param output directory to store temporary diagonal and affinity data
   * @param dimensions the square dimension of the matrices (theses should be the same)
   */
  public static String[] runJob(Path affPath, Path sensitivityPath, Path outputPath, int dimensions, Configuration conf)
    throws IOException, ClassNotFoundException, InterruptedException {
    
    // set up all the job tasks
	System.out.println("You have entered the Diagonalizable Affinity Job");

    conf.setInt(EigencutsKeys.AFFINITY_DIMENSIONS, dimensions);
    Job job = new Job(conf, "DiagonalAffinityJob");
    job.setJarByClass(DiagonalAffinityJob.class);
    //Do not need to set inputFormatClass--Taken care of by MultipleInputs
    
    //Affinity matrix read in as a sequence file and tagged with affPathTaggerMapper.class
    MultipleInputs.addInputPath(job, affPath, SequenceFileInputFormat.class, affPathTaggerMapper.class);
   
    //Sensitivity matrix read in as a sequence file and tagged with sensitivityPathTaggerMapper.class
  	MultipleInputs.addInputPath(job, sensitivityPath, SequenceFileInputFormat.class, sensitivityPathTaggerMapper.class);
  	
  	MultipleOutputs.addNamedOutput(job, "affMatrixTemp", SequenceFileOutputFormat.class, IntWritable.class, VectorWritable.class);

    MultipleOutputs.addNamedOutput(job, "sensitivityValues", SequenceFileOutputFormat.class, IntWritable.class, DoubleWritable.class);
   
    FileOutputFormat.setOutputPath(job, outputPath);
   
    
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(MultiLabelVectorWritable.class);
    
    job.setReducerClass(MatrixDiagonalizeReducer.class);
         
    boolean succeeded = job.waitForCompletion(true);
    if (!succeeded) {
      throw new IllegalStateException("Job failed!");
    }
    
    //dataPaths[0] is the affinity matrix with diagonal values that are not updated
    //dataPaths[1] is the diagonal information needed to determine # of cuts and diagonal values
    //dataPaths[2] is the number of cuts
    String[] pathNamesAndNumCuts = new String[3];
    pathNamesAndNumCuts[0] = "affMatrixTemp-r-00000"; 
    pathNamesAndNumCuts[1] = "sensitivityValues-r-00000";
    pathNamesAndNumCuts[2] = Long.toString(job.getCounters().findCounter(CUTSCOUNTER.NUM_CUTS).getValue());
    
    return pathNamesAndNumCuts;

  }
  
  public static class affPathTaggerMapper
    extends Mapper<IntWritable, VectorWritable, IntWritable, MultiLabelVectorWritable> {
    
    @Override
    protected void map(IntWritable key, VectorWritable row, Context context) 
      throws IOException, InterruptedException {
   

    	System.out.println("You are in rowPathTaggerMapper");
	    System.out.println("Key:" + key);
	    System.out.println("Row:" + row.get());
    	
    	//Tag the vectors coming in to this mapper as 0 for label
	    int[] i = {0};
    	MultiLabelVectorWritable node = new MultiLabelVectorWritable(row.get(), i);
  	 	
   	
    	context.write(key, node);
   
    }
  }
    
  	public static class sensitivityPathTaggerMapper
    extends Mapper<IntWritable, VectorWritable, IntWritable, MultiLabelVectorWritable> {
    
    @Override
    protected void map(IntWritable key, VectorWritable row, Context context) 
      throws IOException, InterruptedException {
    	
    	System.out.println("You are in sensitivityPathTaggerMapper");
	    	System.out.println("Key:" + key);
	    	System.out.println("Row:" + row.get());
	    
	    	//Tag the vectors coming in to this mapper as 1 for label
	    	int[] i = {1};
	    	MultiLabelVectorWritable node = new MultiLabelVectorWritable(row.get(), i);
	  	 	
	   	 	context.write(key, node);
	    	 

    	}
    }
    	
  	public static class MatrixDiagonalizeReducer 
    extends Reducer<IntWritable, MultiLabelVectorWritable, IntWritable, VectorWritable> {
    
  	@SuppressWarnings("rawtypes")
	private MultipleOutputs out;

  	@SuppressWarnings({ "rawtypes", "unchecked" })
  	public void setup(Context context) 
		{

  		out = new MultipleOutputs(context);
  		
		}
 
	@SuppressWarnings("unchecked")
	@Override
    protected void reduce(IntWritable key, Iterable<MultiLabelVectorWritable> values,
      Context context) throws IOException, InterruptedException {
     
    	System.out.println("You are in the reducer");
    	System.out.println("Key:" + key);
    	
       RandomAccessSparseVector aff = null;
       RandomAccessSparseVector sen = null;
    	
    	//Get the appropriate vectors 
    	for(MultiLabelVectorWritable e : values)
    	{
    		System.out.println("Label placement start.");
    		System.out.println("Label placement e:" + e.getLabels()[0]);
    		if(e.getLabels()[0]==0) 	 { aff = new RandomAccessSparseVector(e.getVector()); } 
    		else if(e.getLabels()[0]==1) { sen = new RandomAccessSparseVector(e.getVector()); }
    	}
    	
    	//
    	if(aff==null || sen==null) { return; }
    	
    	System.out.println("affVector:" + aff);
    	System.out.println("senVector:" + sen);
    	Configuration c = context.getConfiguration();
    	int dimensions = c.getInt(EigencutsKeys.AFFINITY_DIMENSIONS, Integer.MAX_VALUE);
    	System.out.println("Dimensions:" + dimensions);
    	
    	Iterator<Vector.Element> itr = sen.iterateNonZero();

    	//Iterate all the nonzeros of the sensitivity matrix
    	while(itr.hasNext() )
    	{
    		Vector.Element e = itr.next();
    		int currentIndex = e.index();
    		//Horizontal diagonal value component is the row and the value in the affinity at the index 
    		out.write("sensitivityValues", key, new DoubleWritable(aff.get(currentIndex)));
    		//Vertical diagonal value component
    		out.write("sensitivityValues", new IntWritable(currentIndex), new DoubleWritable(aff.get(currentIndex)));
    		//Cut the affinity vector at the location
    		aff.setQuick(currentIndex, 0);
    		//increment the counter
    		context.getCounter(CUTSCOUNTER.NUM_CUTS).increment(1);
    	}
    	System.out.println("FinalAffVector:" + aff);
    	out.write("affMatrixTemp",key, new VectorWritable(aff));
    		 
      }
    
    }
}
  

