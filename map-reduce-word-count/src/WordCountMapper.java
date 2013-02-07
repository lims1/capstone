import java.io.IOException;
import java.util.StringTokenizer;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;


public class WordCountMapper extends Mapper<LongWritable,Text,Text,IntWritable> 
{
	private static final IntWritable one = new IntWritable(1);
	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException
	{
		String line = value.toString();
		line = line.replaceAll("[^A-Za-z0-9]", " ");
		StringTokenizer tokenizer = new StringTokenizer(line);
		while(tokenizer.hasMoreTokens())
		{
			Text token = new Text(tokenizer.nextToken());
			context.write(token, one);
		}
	}
	
	
}
