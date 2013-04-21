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
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.math.MultiLabelVectorWritable;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

/**
 * <p>The point of this class is to take all the arrays of sensitivities
 * and convert them to a single matrix. Since there may be many values
 * that, according to their (i, j) coordinates, overlap in the matrix,
 * the "winner" will be determined by whichever value is smaller.</p> 
 */
public class EigencutsSensitivityCutsReducer extends
    Reducer<IntWritable, MultiLabelVectorWritable, IntWritable, VectorWritable> {

  @Override
  protected void reduce(IntWritable key, Iterable<MultiLabelVectorWritable> arr, Context context)
    throws IOException, InterruptedException {
    Configuration conf = context.getConfiguration();

   RandomAccessSparseVector aff = null;
   RandomAccessSparseVector sen = new RandomAccessSparseVector(conf.getInt(EigencutsKeys.AFFINITY_DIMENSIONS, Integer.MAX_VALUE), 100);
	
	//Get the affinityVector first
	for(MultiLabelVectorWritable e : arr)
	{
		if(e.getLabels()[0]==0){
			aff = new RandomAccessSparseVector(e.getVector()); 
		}
		else if(e.getLabels()[0]==1)
		{
			RandomAccessSparseVector tempSen = new RandomAccessSparseVector(e.getVector());
			Iterator<Vector.Element> senItr = tempSen.iterateNonZero();
			while(senItr.hasNext())
			{
				Vector.Element ve = senItr.next();
				if(ve.get() < sen.get(ve.index()))
				{
					sen.setQuick(ve.index(), ve.get());
				}
			}
		}
	}

	if(aff==null ) { return; }
   
	Iterator<Vector.Element> itr = aff.iterateNonZero();
	while(itr.hasNext())
	{
		Vector.Element element = itr.next();
		if(key.get() == element.index()) { continue; }
		
		if(sen.get(element.index())!=0)
		{
			//Setting diagonal
			aff.set(key.get(), aff.get(key.get()) + aff.get(element.index()));
			//Cut element and increment counter
			aff.set(element.index(), 0);
			context.getCounter(EigencutsSensitivityCutsJob.CUTSCOUNTER.NUM_CUTS).increment(1);
		}	
	}
	
	context.write(key, new VectorWritable(aff));
	
  }
}
