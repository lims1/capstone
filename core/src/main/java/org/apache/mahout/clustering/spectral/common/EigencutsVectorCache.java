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

	package org.apache.mahout.clustering.spectral.common;

	import java.io.IOException;
	import java.net.URI;
	import java.util.Arrays;

	import com.google.common.io.Closeables;
	import org.apache.hadoop.conf.Configuration;
	import org.apache.hadoop.filecache.DistributedCache;
	import org.apache.hadoop.fs.FileSystem;
	import org.apache.hadoop.fs.Path;
	import org.apache.hadoop.io.IntWritable;
	import org.apache.hadoop.io.SequenceFile;
	import org.apache.hadoop.io.Writable;
	import org.apache.mahout.common.HadoopUtil;
	import org.apache.mahout.common.iterator.sequencefile.SequenceFileValueIterator;
	import org.apache.mahout.math.Vector;
	import org.apache.mahout.math.VectorWritable;
	import org.slf4j.Logger;
	import org.slf4j.LoggerFactory;


	/**
	 * This class handles reading and writing vectors to the Hadoop
	 * distributed cache. Created as a result of Eigencuts' liberal use
	 * of such functionality, but also due to VectorCache's handling of only
	 * one vector in the Distributedcache.
	 */
	public final class EigencutsVectorCache {

	  private static final Logger log = LoggerFactory.getLogger(VectorCache.class);

	  private EigencutsVectorCache() {
	  }

	  /**
	   * 
	   * @param key SequenceFile key
	   * @param vector Vector to save, to be wrapped as VectorWritable
	   */
	  public static void save(Writable eigenvalueKey,
	                          Vector eigenvalues,
	                          Path eigenvaluePath,
	                          Writable diagonalKey,
	                          Vector diagonal,
	                          Path diagonalPath,
	                          Configuration conf,
	                          boolean overwritePath,
	                          boolean deleteOnExit) throws IOException {
	    
		// set the cache
		DistributedCache.setCacheFiles(new URI[] {eigenvaluePath.toUri(), diagonalPath.toUri()}, conf);
		    
	    
		  
		FileSystem fs = FileSystem.get(eigenvaluePath.toUri(), conf);
	    eigenvaluePath = fs.makeQualified(eigenvaluePath);
	    if (overwritePath) {
	      HadoopUtil.delete(conf, eigenvaluePath);
	    }


	    // set up the writer
	    SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, eigenvaluePath, 
	        IntWritable.class, VectorWritable.class);
	    try {
	      writer.append(eigenvalueKey, new VectorWritable(eigenvalues));
	    } finally {
	      Closeables.closeQuietly(writer);
	    }

	    if (deleteOnExit) {
	      fs.deleteOnExit(eigenvaluePath);
	    }
	   
	    
	    fs = FileSystem.get(diagonalPath.toUri(), conf);
	    diagonalPath = fs.makeQualified(diagonalPath);
	    if (overwritePath) {
	      HadoopUtil.delete(conf, diagonalPath);
	    }
	    
	    
	    //set up the writer again
	    writer = new SequenceFile.Writer(fs, conf, diagonalPath, 
		        IntWritable.class, VectorWritable.class);
		    try {
		      writer.append(diagonalKey, new VectorWritable(diagonal));
		    } finally {
		      Closeables.closeQuietly(writer);
		    }

		    if (deleteOnExit) {
		      fs.deleteOnExit(diagonalPath);
		    }  
	    
	  }
	  
	  /**
	   * Calls the save() method, setting the cache to overwrite any previous
	   * Path and to delete the path after exiting
	   */
	  public static void save(Writable eigenvalueKey, Vector eigenvalues, Path eigenvaluePath, Writable diagonalKey,
              					Vector diagonal, Path diagonalPath, Configuration conf) throws IOException {
	    save(eigenvalueKey,eigenvalues, eigenvaluePath,diagonalKey, diagonal, diagonalPath, conf, true, true);
	  }
	  
	  /**
	   * Loads the vector from {@link DistributedCache}. Returns null if no vector exists.
	   */
	  public static Vector load(int key, Configuration conf) throws IOException {
	    URI[] files = DistributedCache.getCacheFiles(conf);
	    if (files == null || files.length < 1) {
	      return null;
	    }
	    log.info("Files are: {}", Arrays.toString(files));
	    return load(conf, new Path(files[key].getPath()));
	  }
	  
	  /**
	   * Loads a Vector from the specified path. Returns null if no vector exists.
	   */
	  public static Vector load(Configuration conf, Path input) throws IOException {
	    log.info("Loading vector from: {}", input);
	    SequenceFileValueIterator<VectorWritable> iterator =
	        new SequenceFileValueIterator<VectorWritable>(input, true, conf);
	    try {
	      return iterator.next().get();
	    } finally {
	      Closeables.closeQuietly(iterator);
	    }
	  }
	}



