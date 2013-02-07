import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapreduce.Reducer;


public class WordCountReducer extends Reducer<Text, IntWritable, Text, IntWritable> 
{
	public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException
	{
		int count = 0;
		Iterator<IntWritable> iter = values.iterator();
		while(iter.hasNext())
		{
			count += iter.next().get();
		}
		context.write(key,new IntWritable(count));
	}
	
}
