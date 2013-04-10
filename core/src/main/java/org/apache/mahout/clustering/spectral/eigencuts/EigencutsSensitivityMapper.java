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
import java.util.Map;

import com.google.common.collect.Maps;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.clustering.spectral.common.EigencutsVectorCache;
import org.apache.mahout.clustering.spectral.common.VectorCache;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.Functions;

public class EigencutsSensitivityMapper extends
    Mapper<IntWritable, VectorWritable, IntWritable, EigencutsSensitivityNode> {

  private Vector eigenvalues;
  private Vector diagonal;
  private double beta0;
  private double epsilon;
  
  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    Configuration config = context.getConfiguration();
    beta0 = Double.parseDouble(config.get(EigencutsKeys.BETA));
    System.out.println("Loaded beta from mapper:" + Double.toString(beta0));
    epsilon = Double.parseDouble(config.get(EigencutsKeys.EPSILON));
    System.out.println("Loaded epsilon from mapper:" + Double.toString(epsilon));
    // read in the two vectors from the cache
    eigenvalues = EigencutsVectorCache.load(EigencutsKeys.EIGENVALUES_CACHE_INDEX, config);
    System.out.println("Loaded eigenvalues:" + eigenvalues);
    diagonal = EigencutsVectorCache.load(EigencutsKeys.DIAGONAL_CACHE_INDEX, config);
    System.out.println("Loaded diagonal:" + diagonal);
    if (!(eigenvalues instanceof SequentialAccessSparseVector || eigenvalues instanceof DenseVector)) {
      eigenvalues = new SequentialAccessSparseVector(eigenvalues);
    }
    if (!(diagonal instanceof SequentialAccessSparseVector || diagonal instanceof DenseVector)) {
      diagonal = new SequentialAccessSparseVector(diagonal);
    }
  }
  
  @Override
  protected void map(IntWritable row, VectorWritable vw, Context context) 
    throws IOException, InterruptedException {
   
	System.out.println("RowNumber:" + row);
    System.out.println("Vector:" + vw);
    
    // first, does this particular eigenvector even pass the required threshold?
    double eigenvalue = Math.abs(eigenvalues.get(row.get()));
    double betak = -Functions.LOGARITHM.apply(2) / Functions.LOGARITHM.apply(eigenvalue);
    if (eigenvalue >= 1.0 || betak <= epsilon * beta0) {
      // doesn't pass the threshold! quit
    	System.out.println("This eigenvector doesn't pass the required threshold!");
    	System.out.println("Eigenvalue: " +  eigenvalue);
    	System.out.println("betak: " +  betak);
    	System.out.println("epsilon*beta0 = " + epsilon + " * " + beta0 + " =" + epsilon*beta0);
    	if(betak <= epsilon * beta0)
    		System.out.println("betak <= epsilon * beta0 is true");
    	if (eigenvalue >= 1.0)
    		System.out.println("eigenvalue >= 1.0 is true.");
      return;
    }
    
    // go through the vector, performing the calculations
    // sadly, no way to get around n^2 computations      
    // for simplicity purposes non-maximal suppression also not included for now
    Vector ev = vw.get();
    for (int i = 0; i < ev.size(); i++) {
      for (int j = 0; j < ev.size(); j++) {          
        double sij = performSensitivityCalculation(eigenvalue, ev.get(i),
            ev.get(j), diagonal.get(i), diagonal.get(j));
        	EigencutsSensitivityNode e = new EigencutsSensitivityNode(row.get(), j, sij);
        	System.out.println("Row being written out:" + e.getRow());
        	System.out.println("Node being written out: \n" + e);
        	System.out.println("-----------------------------------------------------");
        	context.write(new IntWritable(e.getRow()), e);
      }
    }
    
  }
  
  /**
   * Helper method, performs the actual calculation. Looks something like this:
   *
   * (log(2) / lambda_k * log(lambda_k) * log(lambda_k^beta0 / 2)) * [
   * - (((u_i / sqrt(d_i)) - (u_j / sqrt(d_j)))^2 + (1 - lambda) * 
   *   ((u_i^2 / d_i) + (u_j^2 / d_j))) ]
   */
  private double performSensitivityCalculation(double eigenvalue,
                                               double evi,
                                               double evj,
                                               double diagi,
                                               double diagj) {
    System.out.println("Eigenvalue in Sensitivity Calculation:" + eigenvalue);
    System.out.println("Evi in Sensitivity Calculation:" + evi);
    System.out.println("Evj in Sensitivity Calculation:" + evj);
    System.out.println("Diagi in Sensitivity Calculation:" + diagi);
    System.out.println("Diagj in Sensitivity Calculation:" + diagj);
    
    double firsthalf = Functions.LOGARITHM.apply(2)
        / (eigenvalue * Functions.LOGARITHM.apply(eigenvalue)
           * Functions.LOGARITHM.apply(Functions.POW.apply(eigenvalue, beta0) / 2));
    
    double secondhalf =
        -Functions.POW.apply(evi / Functions.SQRT.apply(diagi) - evj / Functions.SQRT.apply(diagj), 2)
        + (1.0 - eigenvalue) * (Functions.POW.apply(evi, 2) / diagi + Functions.POW.apply(evj, 2) / diagj);
    
    System.out.println(firsthalf + "  x " + secondhalf + " = " + (firsthalf*secondhalf));
    
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
