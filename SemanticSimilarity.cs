namespace MoMi;
class SemanticSimilarity
{
	private List<Blueprint> blueprints;
	private List<string> data;
	
	public SemanticSimilarity(List<Blueprint> blueprints, List<string> data)
	{
		this.blueprints = blueprints;
		this.data = data;
	}
	
	public double[,] ComputeSimilarityMatrix()
	{
		int size = data.Count;
		Console.WriteLine(size);
		
		double[,] similarityMatrix = new double[size, size];
		for (int i = 0; i < size; i++)
		{
			for (int j = 0; j < size; j++)
			{
				//double distance = ComputeLevenshteinDistance(PreProcessData(data[i]), PreProcessData(data[j]));
				double distanceData = ComputeLongestCommonSubsequence(PreProcessData(data[i]), PreProcessData(data[j]));
				double distanceClass = ComputeLongestCommonSubsequence(PreProcessData(blueprints[i].className), PreProcessData(blueprints[j].className));
				double distanceMethod = ComputeLongestCommonSubsequence(PreProcessData(blueprints[i].methodName), PreProcessData(blueprints[i].methodName));
				double distance = distanceClass + distanceMethod + distanceData;
				// double distance = ComputeJaccardSimilarity(data[i], data[j]);
				// double distance = ComputeCosineSimilarity(data[i], data[j]);
				similarityMatrix[i, j] = distance;
				similarityMatrix[j, i] = distance;
			}
		}
		
		return similarityMatrix;
	}
	
	private string PreProcessData(string data)
	{
		string[] words = File.ReadAllLines("./StopWords.txt");
		foreach (string word in words)
		{
			string processedWord = word;
			processedWord = word.Replace("\n", "").Replace("\r", "");
			data = data.Replace(processedWord, "", StringComparison.OrdinalIgnoreCase);
		}
		return data;
	}
	
	private double ComputeLevenshteinDistance(string s, string t)
	{
		double[,] distanceMatrix = new double[s.Length + 1, t.Length + 1];

		for (int i = 0; i <= s.Length; i++)
		{
			distanceMatrix[i, 0] = i;
		}

		for (int j = 0; j <= t.Length; j++)
		{
			distanceMatrix[0, j] = j;
		}

		for (int j = 1; j <= t.Length; j++)
		{
			for (int i = 1; i <= s.Length; i++)
			{
				if (s[i - 1] == t[j - 1])
				{
					distanceMatrix[i, j] = distanceMatrix[i - 1, j - 1];
				}
				else
				{
					distanceMatrix[i, j] = Math.Min(Math.Min(distanceMatrix[i - 1, j] + 1, distanceMatrix[i, j - 1] + 1), distanceMatrix[i - 1, j - 1] + 1);
				}
			}
		}

		return distanceMatrix[s.Length, t.Length];
	}
	
	private double ComputeLongestCommonSubsequence(string s, string t)
	{
		int m = s.Length;
		int n = t.Length;

		int[,] dp = new int[m + 1, n + 1];

		for (int i = 0; i <= m; i++)
		{
			for (int j = 0; j <= n; j++)
			{
				if (i == 0 || j == 0)
				{
					dp[i, j] = 0;
				}
				else if (s[i - 1] == t[j - 1])
				{
					dp[i, j] = dp[i - 1, j - 1] + 1;
				}
				else
				{
					dp[i, j] = Math.Max(dp[i - 1, j], dp[i, j - 1]);
				}
			}
		}

		return dp[m, n];
	}
	
	private double ComputeJaccardSimilarity(string s, string t)
	{
		HashSet<char> set1 = new HashSet<char>(s);
		HashSet<char> set2 = new HashSet<char>(t);

		int intersectionCount = set1.Intersect(set2).Count();
		int unionCount = set1.Count + set2.Count - intersectionCount;

		return (double)intersectionCount / unionCount;
	}
	
	private double ComputeCosineSimilarity(string s, string t)
	{
		List<string> words1 = s.ToLower().Split(' ').ToList();
		List<string> words2 = t.ToLower().Split(' ').ToList();

		List<string> uniqueWords = words1.Union(words2).ToList();

		int[] vector1 = new int[uniqueWords.Count];
		int[] vector2 = new int[uniqueWords.Count];

		for (int i = 0; i < uniqueWords.Count; i++)
		{
			vector1[i] = words1.Count(w => w == uniqueWords[i]);
			vector2[i] = words2.Count(w => w == uniqueWords[i]);
		}

		double dotProduct = 0;
		double magnitude1 = 0;
		double magnitude2 = 0;

		for (int i = 0; i < uniqueWords.Count; i++)
		{
			dotProduct += vector1[i] * vector2[i];
			magnitude1 += Math.Pow(vector1[i], 2);
			magnitude2 += Math.Pow(vector2[i], 2);
		}

		magnitude1 = Math.Sqrt(magnitude1);
		magnitude2 = Math.Sqrt(magnitude2);

		double cosineSimilarity = dotProduct / (magnitude1 * magnitude2);

		return cosineSimilarity;
	}

}